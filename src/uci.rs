//! Minimal-but-correct UCI. Search runs on a worker thread so `stop` is honored mid-search.
//! Castling conversion (cozy-chess king-takes-rook <-> standard UCI) is delegated to
//! cozy_chess::util::{parse_uci_move, display_uci_move}.

use crate::search::{Limits, Searcher};
use crate::tt::TT;
use cozy_chess::util::{display_uci_move, parse_uci_move};
use cozy_chess::Board;
use std::io::BufRead;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// Build `n` Searchers sharing one TT and one stop flag (Lazy SMP) -- thread 0 is main
/// (prints info/returns bestmove, bumps TT age); the rest search silently.
fn build_pool(hash_mb: usize, n: usize, stop: &Arc<AtomicBool>) -> (Arc<TT>, Vec<Arc<Mutex<Searcher>>>) {
    let tt = Arc::new(TT::new(hash_mb));
    let pool = (0..n.max(1))
        .map(|i| {
            let mut s = Searcher::for_thread(tt.clone(), stop.clone(), i == 0);
            s.thread_id = i;
            s.silent = i != 0;
            Arc::new(Mutex::new(s))
        })
        .collect();
    (tt, pool)
}

fn parse_fen(fields: &[&str]) -> Result<Board, String> {
    let fen = match fields.len() {
        4 => format!("{} 0 1", fields.join(" ")),
        6 => fields.join(" "),
        n => return Err(format!("expected 4 or 6 fields, got {n}")),
    };
    Board::from_fen(&fen, false).map_err(|e| e.to_string())
}

pub fn uci_loop() {
    let mut hash_mb: usize = 64;
    let mut n_threads: usize = 1;
    let stop = Arc::new(AtomicBool::new(false));
    let (mut tt, mut pool) = build_pool(hash_mb, n_threads, &stop);
    let mut board = Board::startpos();
    let mut position_valid = true;
    let mut game_hist: Vec<u64> = Vec::new();
    // this side's own root scores across the game, for volatility-aware time management
    let mut eval_hist_game: Vec<i32> = Vec::new();
    let book = crate::book::Book::load().expect("book failed legality walk");
    // DEFAULT FLIPPED TO FALSE, 2026-09-09. Every SPRT this engine has ever run passes
    // `option.OwnBook=false`, so `true` was the one configuration that had never been measured
    // -- and it was the one an operator using defaults would get. The match is a CCRL-style
    // setup where the tester supplies the openings, which is exactly the case where an internal
    // repertoire is at best redundant and at worst steers into a line the book stops in but the
    // search has never had to hold. Still available on request; just no longer the default.
    let mut own_book = false;
    // Set by `ucinewgame`, cleared by the next `go`. Without it the first search of a new
    // game pushes the LAST game's final score into the fresh volatility history.
    let mut new_game = false;
    let mut adaptive_time = false;
    let mut move_overhead: u128 = 10;
    let mut book_seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE)
        | 1;

    let mut workers: Vec<std::thread::JoinHandle<()>> = Vec::new();
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.first().copied() {
            Some("uci") => {
                println!("id name Patzer 0.2");
                println!("id author Hubert Lipski");
                println!("option name Hash type spin default 64 min 1 max 4096");
                println!("option name Threads type spin default 1 min 1 max 256");
                println!("option name OwnBook type check default false");
                println!("option name AdaptiveTime type check default false");
                println!("option name Move Overhead type spin default 10 min 0 max 5000");
                println!("option name EvalFile type string default <empty>");
                println!("option name PolicyFile type string default <empty>");
                #[cfg(feature = "tune")]
                for (n, d, lo, hi) in crate::search::tune::PARAMS {
                    println!("option name {n} type spin default {d} min {lo} max {hi}");
                }
                println!("uciok");
            }
            Some("isready") => println!("readyok"),
            Some("setoption") => {
                // setoption name Hash value N
                if let (Some(ni), Some(vi)) = (
                    tokens.iter().position(|&t| t == "name"),
                    tokens.iter().position(|&t| t == "value"),
                ) {
                    if tokens.get(ni + 1).map(|s| s.eq_ignore_ascii_case("hash")) == Some(true) {
                        if let Some(mb) = tokens.get(vi + 1).and_then(|v| v.parse::<usize>().ok()) {
                            hash_mb = mb;
                            let (new_tt, new_pool) = build_pool(hash_mb, n_threads, &stop);
                            tt = new_tt;
                            pool = new_pool;
                        }
                    } else if tokens.get(ni + 1).map(|s| s.eq_ignore_ascii_case("threads"))
                        == Some(true)
                    {
                        if let Some(n) = tokens.get(vi + 1).and_then(|v| v.parse::<usize>().ok()) {
                            n_threads = n.max(1);
                            let (new_tt, new_pool) = build_pool(hash_mb, n_threads, &stop);
                            tt = new_tt;
                            pool = new_pool;
                        }
                    } else if tokens.get(ni + 1).map(|s| s.eq_ignore_ascii_case("ownbook"))
                        == Some(true)
                    {
                        own_book = tokens.get(vi + 1).map(|v| v.eq_ignore_ascii_case("true"))
                            == Some(true);
                    } else if tokens.get(ni + 1).map(|s| s.eq_ignore_ascii_case("adaptivetime"))
                        == Some(true)
                    {
                        adaptive_time = tokens.get(vi + 1).map(|v| v.eq_ignore_ascii_case("true"))
                            == Some(true);
                    } else if tokens.get(ni + 1).map(|s| s.eq_ignore_ascii_case("move")) == Some(true)
                        && tokens.get(ni + 2).map(|s| s.eq_ignore_ascii_case("overhead"))
                            == Some(true)
                    {
                        // "Move Overhead" has a space, so it is two tokens after `name`.
                        if let Some(v) = tokens.get(vi + 1).and_then(|v| v.parse::<u128>().ok()) {
                            move_overhead = v.min(5000);
                        }
                    } else if tokens.get(ni + 1).map(|s| s.eq_ignore_ascii_case("evalfile"))
                        == Some(true)
                    {
                        if let Some(path) = tokens.get(vi + 1) {
                            match crate::nnue::load_global(path) {
                                Ok(()) => println!("info string NNUE loaded: {path}"),
                                Err(e) => println!("info string NNUE load failed: {e}"),
                            }
                        }
                    } else if tokens.get(ni + 1).map(|s| s.eq_ignore_ascii_case("policyfile"))
                        == Some(true)
                    {
                        if let Some(path) = tokens.get(vi + 1) {
                            match crate::policy::load_global(path) {
                                Ok(()) => println!("info string policy loaded: {path}"),
                                Err(e) => println!("info string policy load failed: {e}"),
                            }
                        }
                    } else {
                        // Search-parameter tunables. Only exist under `--features tune`; in a
                        // release build an unknown option is ignored exactly as before.
                        #[cfg(feature = "tune")]
                        if let (Some(n), Some(v)) =
                            (tokens.get(ni + 1), tokens.get(vi + 1).and_then(|v| v.parse::<i32>().ok()))
                        {
                            if crate::search::tune::set(n, v) {
                                println!("info string tune {n} = {v}");
                            }
                        }
                    }
                }
            }
            Some("ucinewgame") => {
                stop.store(true, Ordering::Relaxed);
                for h in workers.drain(..) {
                    let _ = h.join();
                }
                tt.clear();
                for searcher in &pool {
                    if let Ok(mut searcher) = searcher.lock() {
                        searcher.new_game();
                    }
                }
                // Per-GAME state, so it is cleared HERE and not in `position` -- see below.
                eval_hist_game.clear();
                new_game = true;
            }
            Some("position") => {
                let mut idx = 1;
                let mut next_board;
                if tokens.get(idx) == Some(&"startpos") {
                    next_board = Board::startpos();
                    idx += 1;
                } else if tokens.get(idx) == Some(&"fen") {
                    let end = tokens
                        .iter()
                        .position(|&t| t == "moves")
                        .unwrap_or(tokens.len());
                    match parse_fen(&tokens[idx + 1..end]) {
                        Ok(b) => next_board = b,
                        Err(e) => {
                            println!("info string bad fen: {e}");
                            position_valid = false;
                            game_hist.clear();
                            continue;
                        }
                    }
                    idx = end;
                } else {
                    println!("info string bad position command");
                    position_valid = false;
                    game_hist.clear();
                    continue;
                }
                position_valid = true;
                let mut next_hist = Vec::new();
                // NOT eval_hist_game. A GUI sends `position` before every single `go`, so
                // clearing it here capped it at one entry for ever, `recent_volatility()`
                // (which needs three) could only ever return None, and VOL_TM was a no-op
                // however it was gated. That is why the 2912-game `voltm` campaign row read
                // -4.89 +/- 8.52: it measured a binary against itself. Bug present since
                // a040d63, the commit that added the feature. See AUDIT_SEARCH_20260913.md S1.
                // `ucinewgame` clears it.
                if tokens.get(idx) == Some(&"moves") {
                    for mv_str in &tokens[idx + 1..] {
                        // `parse_uci_move` PARSES; it does not validate. Its whole body is a
                        // string parse plus the king-takes-rook castling conversion, so it
                        // returns Ok for any well-formed <sq><sq>[promo] -- including moves
                        // that are illegal in this position. `play_unchecked` then executes
                        // them, and two of the resulting states are fatal, both measured:
                        //
                        //   moves e2e4 e7e5 e2e5   -> panic "Missing piece on move's from
                        //                             square" ON THE MAIN THREAD: process dies.
                        //   moves ... Qxe8 (king)  -> board with no king; the SEARCH thread
                        //                             panics "No king was found", main lives,
                        //                             no bestmove is ever sent and the GUI hangs.
                        //
                        // A conforming GUI never sends these, which is exactly why it went
                        // unnoticed. `is_legal` is the same guard the book probe and
                        // `packed_to_move` already use to make an illegal move impossible.
                        match parse_uci_move(&next_board, mv_str) {
                            Ok(mv) if !next_board.is_legal(mv) => {
                                println!("info string illegal move {mv_str}");
                                position_valid = false;
                                break;
                            }
                            Ok(mv) => {
                                next_hist.push(crate::search::repetition_hash(&next_board));
                                next_board.play_unchecked(mv);
                            }
                            Err(e) => {
                                println!("info string bad move {mv_str}: {e}");
                                position_valid = false;
                                break;
                            }
                        }
                    }
                }
                if position_valid {
                    board = next_board;
                    game_hist = next_hist;
                } else {
                    game_hist.clear();
                }
            }
            Some("go") => {
                if !position_valid {
                    println!("info string no valid position");
                    println!("bestmove 0000");
                    continue;
                }
                // book probe: instant reply while the position is in repertoire
                if own_book && !tokens.contains(&"infinite") {
                    if let Some(mv) = book.probe(board.hash(), &mut book_seed) {
                        if board.is_legal(mv) {
                            println!("bestmove {}", display_uci_move(&board, mv));
                            continue;
                        }
                    }
                }
                let mut limits = Limits::default();
                limits.adaptive_time = adaptive_time;
                limits.move_overhead = move_overhead;
                let mut it = tokens[1..].iter();
                while let Some(&tok) = it.next() {
                    let num =
                        |it: &mut std::slice::Iter<&str>| it.next().and_then(|v| v.parse().ok());
                    match tok {
                        "depth" => limits.depth = num(&mut it).map(|v: i64| v as i32),
                        "nodes" => limits.nodes = num(&mut it).map(|v: i64| v as u64),
                        "movetime" => limits.movetime = num(&mut it).map(|v: i64| v as u128),
                        "wtime" => limits.wtime = num(&mut it).map(|v: i64| v.max(1) as u128),
                        "btime" => limits.btime = num(&mut it).map(|v: i64| v.max(1) as u128),
                        "winc" => limits.winc = num(&mut it).map(|v: i64| v.max(0) as u128),
                        "movestogo" => limits.movestogo = num(&mut it).map(|v: i64| v.max(1) as u32),
                        "binc" => limits.binc = num(&mut it).map(|v: i64| v.max(0) as u128),
                        "infinite" => limits.infinite = true,
                        _ => {}
                    }
                }
                let had_prev = !workers.is_empty();
                for h in workers.drain(..) {
                    let _ = h.join(); // previous search must have printed its bestmove
                }
                // Record the previous search's root score for volatility-aware time
                // management. Safe here and only here: the join above guarantees that search
                // has finished, so no extra synchronisation is needed. These are all THIS
                // side's own scores, from its own perspective, which is exactly the per-side
                // sequence the AUC 0.81 measurement was made on.
                if had_prev && !new_game {
                    if let Ok(s0) = pool[0].lock() {
                        eval_hist_game.push(s0.last_score);
                    }
                }
                new_game = false;
                stop.store(false, Ordering::Relaxed); // reset once, before any thread starts
                for (i, s) in pool.iter().enumerate() {
                    let s = s.clone();
                    let board = board.clone();
                    let hist = game_hist.clone();
                    let evh = eval_hist_game.clone();
                    let limits = limits.clone();
                    let is_main = i == 0;
                    let worker_stop = stop.clone();
                    workers.push(std::thread::spawn(move || {
                        let mut s = match s.try_lock() {
                            Ok(s) => s,
                            Err(_) => {
                                if is_main {
                                    println!("info string already searching");
                                }
                                return;
                            }
                        };
                        s.game_hist = hist;
                        s.eval_hist_game = evh;
                        let best = s.think(&board, &limits);
                        if is_main {
                            // The main thread has decided, so nothing a helper still finds can
                            // be used. Without this they run on to their OWN limits, which under
                            // a real clock are far apart (soft = t/30, hard = t/6): measured
                            // 1.48 CPU-SECONDS burned AFTER `bestmove` at Threads=4, 60+0.6.
                            // That is CPU spent during the opponent's turn, and work the next
                            // `go` must join() before it can start searching. Invisible to the
                            // bench signature, which is fixed-depth and single-threaded.
                            // See AUDIT_SEARCH_20260913.md S2.
                            worker_stop.store(true, Ordering::Relaxed);
                        }
                        if is_main {
                            match best {
                                Some(mv) => println!("bestmove {}", display_uci_move(&board, mv)),
                                None => {
                                    // 2026-09-11 (readiness audit R3): `go depth 0` on the START
                                    // POSITION emitted `bestmove 0000` with 20 legal moves on the
                                    // board -- measured. A stop landing before depth 1 completes
                                    // takes the same path. Some GUIs score that as a forfeit, so
                                    // fall back to a legal move; 0000 only when there is none.
                                    let mut fallback = None;
                                    board.generate_moves(|pm| {
                                        fallback = pm.into_iter().next();
                                        true
                                    });
                                    match fallback {
                                        Some(mv) => {
                                            println!("bestmove {}", display_uci_move(&board, mv))
                                        }
                                        None => println!("bestmove 0000"),
                                    }
                                }
                            }
                        }
                    }));
                }
            }
            Some("stop") => stop.store(true, Ordering::Relaxed),
            Some("quit") => break,
            _ => {}
        }
    }
    stop.store(true, Ordering::Relaxed);
    for h in workers.drain(..) {
        let _ = h.join(); // let the search print bestmove before the process exits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_field_fen_gets_default_move_counters() {
        let short = parse_fen(&["8/8/8/8/8/8/4k3/7K", "w", "-", "-"]).unwrap();
        let full = parse_fen(&["8/8/8/8/8/8/4k3/7K", "w", "-", "-", "0", "1"]).unwrap();
        assert_eq!(short, full);
    }

    /// The trap the `is_legal` guard in the move loop exists for. If cozy ever starts
    /// validating in `parse_uci_move`, this test fails and the comment there becomes wrong --
    /// which is the point: the guard is justified by this behaviour, so the behaviour is
    /// pinned. Both moves below are well-formed strings and illegal on the board.
    #[test]
    fn parse_uci_move_accepts_illegal_moves_so_we_must_check() {
        let b = Board::from_fen(
            "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", false).unwrap();
        let blocked = parse_uci_move(&b, "a1a8").expect("parse_uci_move rejected a1a8");
        assert!(!b.is_legal(blocked), "a1a8 is blocked by the a2 pawn");
        let empty_from = parse_uci_move(&b, "e2e5").expect("parse_uci_move rejected e2e5");
        assert!(!b.is_legal(empty_from), "e2 is empty after e2e4");
    }

    #[test]
    fn malformed_fen_is_rejected() {
        assert!(parse_fen(&["not-a-board", "w", "-", "-"]).is_err());
        assert!(parse_fen(&["8/8/8/8/8/8/4k3/7K", "w", "-"]).is_err());
    }
}
