//! Stage 2a — self-play training data. Fixed 32-byte records; the pack format is verified by the
//! `datagate` round-trip gate (pack -> FEN -> reparse -> cozy zobrist hash must match exactly, so
//! any castling/EP convention mistake fails loudly instead of poisoning the dataset).
//!
//! Record layout (little-endian, 32 bytes):
//!   occ: u64                — occupancy bitboard
//!   pieces: [u8; 16]        — 4-bit codes, ascending-square order of occ bits, low nibble first;
//!                             code = piece(0..5, pawn..king) | color<<3 (0 white, 1 black)
//!   stm: u8                 — 0 white, 1 black
//!   castle: u8              — bit0 K, bit1 Q, bit2 k, bit3 q (standard chess only)
//!   ep: u8                  — en-passant file 0-7, 0xFF none
//!   score: i16              — search score, side-to-move perspective, cp
//!   wdl: u8                 — game result from side-to-move perspective: 0 loss, 1 draw, 2 win
//!   best_move: u16          — packed from | to<<6 | promo<<12 (same packing as the TT)

use crate::search::{Limits, Searcher};
use crate::tt::pack;
use cozy_chess::{Board, Color, GameStatus};
use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub const RECORD_SIZE: usize = 32;

const NODES_PER_MOVE: u64 = 5_000;
const OPENING_PLIES: usize = 8;
const OPENING_FILTER_CP: i32 = 200;
const SKIP_PLIES: usize = 10; // don't record until this many plies into the game
const MAX_SCORE_CP: i32 = 2_000; // decided positions teach nothing
const RESIGN_CP: i32 = 2_500;
const RESIGN_PLIES: u32 = 4;
const DRAW_CP: i32 = 8;
const DRAW_PLIES: u32 = 10;
const DRAW_MIN_PLY: usize = 100;
const MAX_PLIES: usize = 400;

// ---------------- pack / unpack ----------------

pub fn pack_board(board: &Board) -> [u8; 27] {
    let mut out = [0u8; 27];
    let occ = board.occupied();
    out[..8].copy_from_slice(&occ.0.to_le_bytes());
    for (i, sq) in occ.into_iter().enumerate() {
        let p = board.piece_on(sq).unwrap() as u8;
        let c = board.color_on(sq).unwrap() as u8;
        let code = p | (c << 3);
        out[8 + i / 2] |= code << (4 * (i % 2));
    }
    out[24] = board.side_to_move() as u8;
    let mut castle = 0u8;
    let wr = board.castle_rights(Color::White);
    let br = board.castle_rights(Color::Black);
    if wr.short.is_some() {
        castle |= 1;
    }
    if wr.long.is_some() {
        castle |= 2;
    }
    if br.short.is_some() {
        castle |= 4;
    }
    if br.long.is_some() {
        castle |= 8;
    }
    out[25] = castle;
    out[26] = board.en_passant().map(|f| f as u8).unwrap_or(0xFF);
    out
}

pub fn unpack_to_fen(buf: &[u8; 27]) -> String {
    let occ = u64::from_le_bytes(buf[..8].try_into().unwrap());
    let mut grid = [[None::<(u8, u8)>; 8]; 8]; // [rank][file] -> (piece, color)
    let mut i = 0usize;
    for s in 0..64u32 {
        if occ >> s & 1 == 1 {
            let code = (buf[8 + i / 2] >> (4 * (i % 2))) & 0xF;
            grid[(s / 8) as usize][(s % 8) as usize] = Some((code & 7, code >> 3));
            i += 1;
        }
    }
    let mut placement = String::new();
    for rank in (0..8).rev() {
        let mut empty = 0;
        for file in 0..8 {
            match grid[rank][file] {
                Some((p, c)) => {
                    if empty > 0 {
                        placement.push_str(&empty.to_string());
                        empty = 0;
                    }
                    let ch = b"pnbrqk"[p as usize] as char;
                    placement.push(if c == 0 { ch.to_ascii_uppercase() } else { ch });
                }
                None => empty += 1,
            }
        }
        if empty > 0 {
            placement.push_str(&empty.to_string());
        }
        if rank > 0 {
            placement.push('/');
        }
    }
    let stm = if buf[24] == 0 { "w" } else { "b" };
    let mut castle = String::new();
    for (bit, ch) in [(1, 'K'), (2, 'Q'), (4, 'k'), (8, 'q')] {
        if buf[25] & bit != 0 {
            castle.push(ch);
        }
    }
    if castle.is_empty() {
        castle.push('-');
    }
    let ep = if buf[26] == 0xFF {
        "-".to_string()
    } else {
        // ep target square is behind the double-pushed pawn: rank 6 when white is to move
        // (black just pushed), rank 3 when black is to move
        let rank = if buf[24] == 0 { '6' } else { '3' };
        format!("{}{}", (b'a' + buf[26]) as char, rank)
    };
    format!("{placement} {stm} {castle} {ep} 0 1")
}

// ---------------- gate: pack round-trip ----------------

pub fn datagate() {
    println!("GATE datagen — pack->FEN->reparse must reproduce cozy's zobrist hash exactly");
    let mut seed = 0xDA7A_6A7Eu64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut checked = 0usize;
    let mut with_castle = 0usize;
    let mut with_ep = 0usize;
    while checked < 10_000 {
        // random walk of random length so we cover openings, middlegames, endgames
        let mut board = Board::startpos();
        let len = 4 + (rng() % 120) as usize;
        for _ in 0..len {
            let mut moves = Vec::new();
            board.generate_moves(|pm| {
                for mv in pm {
                    moves.push(mv);
                }
                false
            });
            if moves.is_empty() {
                break;
            }
            board.play_unchecked(moves[(rng() % moves.len() as u64) as usize]);
            if board.status() != GameStatus::Ongoing {
                break;
            }

            let packed = pack_board(&board);
            let fen = unpack_to_fen(&packed);
            let rebuilt = match Board::from_fen(&fen, false) {
                Ok(b) => b,
                Err(e) => {
                    println!("  FAIL: unpacked FEN unparsable: {fen} ({e})");
                    println!("  => GATE FAIL");
                    std::process::exit(1);
                }
            };
            if rebuilt.hash() != board.hash() {
                println!("  FAIL: hash mismatch\n  original: {board}\n  rebuilt:  {rebuilt}");
                println!("  => GATE FAIL");
                std::process::exit(1);
            }
            if packed[25] != 0 {
                with_castle += 1;
            }
            if packed[26] != 0xFF {
                with_ep += 1;
            }
            checked += 1;
            if checked >= 10_000 {
                break;
            }
        }
    }
    println!(
        "  {checked} positions round-tripped exactly ({with_castle} w/ castling rights, {with_ep} w/ ep)"
    );
    println!("  => GATE PASS");
}

// ---------------- gate: dataset sanity ----------------

pub fn datacheck(path: &str) {
    println!("DATACHECK {path}");
    let data = std::fs::read(path).expect("cannot read dataset");
    assert!(
        data.len() % RECORD_SIZE == 0,
        "file size {} not a multiple of {RECORD_SIZE}",
        data.len()
    );
    let n = data.len() / RECORD_SIZE;
    let mut wdl_cnt = [0u64; 3];
    let mut score_sum_by_wdl = [0i64; 3];
    let mut hist = [0u64; 9]; // score buckets
    let mut parse_fail = 0u64;
    let step = (n / 2_000).max(1); // spot-parse a sample, count everything
    for i in 0..n {
        let r = &data[i * RECORD_SIZE..(i + 1) * RECORD_SIZE];
        let score = i16::from_le_bytes([r[27], r[28]]) as i32;
        let wdl = r[29] as usize;
        assert!(wdl < 3, "bad wdl at record {i}");
        wdl_cnt[wdl] += 1;
        score_sum_by_wdl[wdl] += score as i64;
        let b = ((score.clamp(-2000, 2000) + 2000) / 500).min(8) as usize;
        hist[b] += 1;
        if i % step == 0 {
            let pos: [u8; 27] = r[..27].try_into().unwrap();
            if Board::from_fen(&unpack_to_fen(&pos), false).is_err() {
                parse_fail += 1;
            }
        }
    }
    println!("  records: {n}");
    println!(
        "  wdl (stm view): L {} / D {} / W {}",
        wdl_cnt[0], wdl_cnt[1], wdl_cnt[2]
    );
    let mean = |i: usize| score_sum_by_wdl[i] as f64 / wdl_cnt[i].max(1) as f64;
    println!(
        "  mean score by wdl: L {:.0} / D {:.0} / W {:.0}  (must be monotone increasing)",
        mean(0),
        mean(1),
        mean(2)
    );
    println!("  score hist [-2000..2000, 500cp bins]: {hist:?}");
    println!("  sampled positions failing FEN parse: {parse_fail}");
    let ok = parse_fail == 0 && mean(0) < mean(1) && mean(1) < mean(2);
    println!("  => {}", if ok { "GATE PASS" } else { "GATE FAIL" });
    if !ok {
        std::process::exit(1);
    }
}

// ---------------- static-eval dump (baseline for the trainer's gate 2b) ----------------

/// For each record, unpack the position and write the engine's static eval (stm cp) as i16 LE.
/// Uses the same unpack path the round-trip gate verifies — no eval logic is ported to Python.
pub fn evalbin(in_path: &str, out_path: &str) {
    let data = std::fs::read(in_path).expect("cannot read dataset");
    assert!(data.len() % RECORD_SIZE == 0, "bad record size");
    let n = data.len() / RECORD_SIZE;
    let mut out = Vec::with_capacity(n * 2);
    for i in 0..n {
        let pos: [u8; 27] = data[i * RECORD_SIZE..i * RECORD_SIZE + 27].try_into().unwrap();
        let board = Board::from_fen(&unpack_to_fen(&pos), false).expect("unpack failed");
        let e = crate::eval::evaluate(&board).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
        out.extend_from_slice(&e.to_le_bytes());
    }
    std::fs::write(out_path, &out).expect("cannot write evals");
    println!("evalbin: {n} static evals -> {out_path}");
}

// ---------------- generation ----------------

struct GameRecord {
    pos: [u8; 27],
    score: i16,
    best: u16,
    stm: Color,
}

pub fn datagen(games: usize, out_path: &str, base_seed: u64, threads: usize) {
    run_datagen(games, out_path, base_seed, threads, None);
}

/// Same self-play pipeline, but each game starts from a position drawn from
/// `positions_path` (one FEN per line) instead of a random balanced opening --
/// used to seed datagen with curated "hard" positions (e.g. mine_hard_positions.py
/// output) so training data covers tactically sharp middlegames the random
/// walk rarely reaches.
pub fn datagen_seeded(games: usize, out_path: &str, base_seed: u64, threads: usize, positions_path: &str) {
    let text = std::fs::read_to_string(positions_path).expect("cannot read positions file");
    let positions: Vec<Board> = text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Board::from_fen(l.trim(), false).expect("bad FEN in positions file"))
        .collect();
    assert!(!positions.is_empty(), "positions file is empty");
    eprintln!("loaded {} seed positions from {positions_path}", positions.len());
    run_datagen(games, out_path, base_seed, threads, Some(positions));
}

fn run_datagen(
    games: usize,
    out_path: &str,
    base_seed: u64,
    threads: usize,
    seed_positions: Option<Vec<Board>>,
) {
    let seed_positions = &seed_positions;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(out_path)
        .expect("cannot open output");
    let writer = Mutex::new(std::io::BufWriter::new(file));
    let games_done = AtomicUsize::new(0);
    let positions = AtomicUsize::new(0);
    let started = std::time::Instant::now();

    std::thread::scope(|scope| {
        for t in 0..threads {
            let writer = &writer;
            let games_done = &games_done;
            let positions = &positions;
            let seed_positions = seed_positions;
            scope.spawn(move || {
                let mut seed = (base_seed ^ (t as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
                let mut rng = move || {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    seed
                };
                let mut searcher = Searcher::new(16);
                searcher.silent = true;

                loop {
                    let g = games_done.fetch_add(1, Ordering::Relaxed);
                    if g >= games {
                        break;
                    }

                    // --- starting position: curated seed if provided, else random balanced opening ---
                    // seeded mode is fully deterministic given a position, so round-robin by game
                    // index (not random sampling) guarantees full coverage with zero duplicate games
                    let board0 = if let Some(pos_list) = seed_positions {
                        pos_list[g % pos_list.len()].clone()
                    } else {
                        loop {
                        let mut b = Board::startpos();
                        let mut ok = true;
                        for _ in 0..OPENING_PLIES {
                            let mut moves = Vec::new();
                            b.generate_moves(|pm| {
                                for mv in pm {
                                    moves.push(mv);
                                }
                                false
                            });
                            if moves.is_empty() {
                                ok = false;
                                break;
                            }
                            b.play_unchecked(moves[(rng() % moves.len() as u64) as usize]);
                        }
                        if !ok || b.status() != GameStatus::Ongoing {
                            continue;
                        }
                        searcher.tt.clear();
                        searcher.game_hist.clear();
                        searcher.think(
                            &b,
                            &Limits {
                                nodes: Some(NODES_PER_MOVE),
                                ..Default::default()
                            },
                        );
                        if searcher.last_score.abs() <= OPENING_FILTER_CP {
                            break b;
                        }
                        }
                    };

                    // --- play the game ---
                    let mut board = board0;
                    let mut hist: Vec<u64> = Vec::new();
                    let mut counts: std::collections::HashMap<u64, u32> =
                        std::collections::HashMap::new();
                    let mut records: Vec<GameRecord> = Vec::with_capacity(128);
                    let mut ply = OPENING_PLIES;
                    let mut resign_run = 0u32;
                    let mut draw_run = 0u32;
                    // result from White's POV: 2 win, 1 draw, 0 loss
                    let result_white: u8;
                    searcher.tt.clear();

                    loop {
                        match board.status() {
                            GameStatus::Won => {
                                // side to move is mated
                                result_white = if board.side_to_move() == Color::White {
                                    0
                                } else {
                                    2
                                };
                                break;
                            }
                            GameStatus::Drawn => {
                                result_white = 1;
                                break;
                            }
                            GameStatus::Ongoing => {}
                        }
                        let c = counts.entry(board.hash()).or_insert(0);
                        *c += 1;
                        if *c >= 3 || board.halfmove_clock() >= 100 || ply >= MAX_PLIES {
                            result_white = 1;
                            break;
                        }

                        searcher.game_hist = hist.clone();
                        let mv = match searcher.think(
                            &board,
                            &Limits {
                                nodes: Some(NODES_PER_MOVE),
                                ..Default::default()
                            },
                        ) {
                            Some(mv) => mv,
                            None => {
                                result_white = 1; // should be unreachable in Ongoing
                                break;
                            }
                        };
                        let score = searcher.last_score;
                        let stm = board.side_to_move();

                        // adjudication
                        if score.abs() >= RESIGN_CP {
                            resign_run += 1;
                            if resign_run >= RESIGN_PLIES {
                                let stm_winning = score > 0;
                                result_white = if (stm == Color::White) == stm_winning {
                                    2
                                } else {
                                    0
                                };
                                break;
                            }
                        } else {
                            resign_run = 0;
                        }
                        if score.abs() <= DRAW_CP && ply >= DRAW_MIN_PLY {
                            draw_run += 1;
                            if draw_run >= DRAW_PLIES {
                                result_white = 1;
                                break;
                            }
                        } else {
                            draw_run = 0;
                        }

                        // record filter: quiet, undecided, past the random opening
                        let is_capture = board.color_on(mv.to).is_some()
                            && board.color_on(mv.to) != Some(stm);
                        let in_check = !board.checkers().is_empty();
                        if ply >= SKIP_PLIES
                            && !in_check
                            && !is_capture
                            && mv.promotion.is_none()
                            && score.abs() < MAX_SCORE_CP
                        {
                            records.push(GameRecord {
                                pos: pack_board(&board),
                                score: score as i16,
                                best: pack(mv),
                                stm,
                            });
                        }

                        hist.push(board.hash());
                        board.play_unchecked(mv);
                        ply += 1;
                    }

                    // --- backfill WDL and write ---
                    let mut buf = Vec::with_capacity(records.len() * RECORD_SIZE);
                    for r in &records {
                        let wdl = if r.stm == Color::White {
                            result_white
                        } else {
                            2 - result_white
                        };
                        buf.extend_from_slice(&r.pos);
                        buf.extend_from_slice(&r.score.to_le_bytes());
                        buf.push(wdl);
                        buf.extend_from_slice(&r.best.to_le_bytes());
                    }
                    positions.fetch_add(records.len(), Ordering::Relaxed);
                    {
                        let mut w = writer.lock().unwrap();
                        w.write_all(&buf).unwrap();
                    }

                    let done = g + 1;
                    if done % 200 == 0 {
                        let secs = started.elapsed().as_secs_f64();
                        let p = positions.load(Ordering::Relaxed);
                        eprintln!(
                            "  {done}/{games} games | {p} positions | {:.1} games/s | {:.0} pos/s",
                            done as f64 / secs,
                            p as f64 / secs
                        );
                    }
                }
            });
        }
    });

    writer.lock().unwrap().flush().unwrap();
    let secs = started.elapsed().as_secs_f64();
    let p = positions.load(Ordering::Relaxed);
    println!(
        "DONE: {games} games, {p} positions, {:.0}s ({:.0} pos/s) -> {out_path}",
        secs,
        p as f64 / secs
    );
}
