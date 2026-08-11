//! patzer — Stage 1: classical alpha-beta engine over verified movegen.
//! Subcommands are the gates: `perft` (exact movegen truth), `mates` (self-verifying mate walker:
//! the harness PROVES the engine's mate claims by exhaustive walk, trusting only perft-verified
//! movegen — no FEN lore taken on faith), `bench` (fixed-depth node-count signature = functional
//! change detector), `soak` (self-play stability: every move legality-checked, no panics).
//! No subcommand => UCI.

mod book;
mod datagen;
mod eval;
mod nnue;
mod policy;
mod pesto;
mod search;
mod tt;
mod uci;

use cozy_chess::{Board, GameStatus, Move};
use search::{Limits, Searcher};

fn main() {
    // net for subcommands (bench/soak/datagen with NNUE); UCI uses the EvalFile option
    if let Ok(path) = std::env::var("PATZER_EVALFILE") {
        nnue::load_global(&path).expect("PATZER_EVALFILE load failed");
    }
    if let Ok(path) = std::env::var("PATZER_POLICYFILE") {
        policy::load_global(&path).expect("PATZER_POLICYFILE load failed");
    }
    match std::env::args().nth(1).as_deref() {
        Some("perft") => perft_gate(),
        Some("mates") => mates_gate(),
        Some("see") => see_gate(),
        Some("book") => book_gate(),
        Some("bench") => bench(),
        Some("soak") => soak(),
        Some("genbook") => genbook(
            std::env::args()
                .nth(2)
                .and_then(|n| n.parse().ok())
                .unwrap_or(64),
        ),
        Some("datagate") => datagen::datagate(),
        Some("policydump") => policydump(
            &std::env::args().nth(2).expect("usage: policydump <data.bin> <out.txt> (needs PATZER_EVALFILE+PATZER_POLICYFILE)"),
            &std::env::args().nth(3).expect("usage: policydump <data.bin> <out.txt>"),
        ),
        Some("nnueinc") => nnueinc(
            &std::env::args().nth(2).expect("usage: nnueinc <net.nnue>"),
        ),
        Some("nnuegate") => nnuegate(
            &std::env::args().nth(2).expect("usage: nnuegate <data.bin> <net.nnue> <ref.i32>"),
            &std::env::args().nth(3).expect("usage: nnuegate <data.bin> <net.nnue> <ref.i32>"),
            &std::env::args().nth(4).expect("usage: nnuegate <data.bin> <net.nnue> <ref.i32>"),
        ),
        Some("evalbin") => datagen::evalbin(
            &std::env::args().nth(2).expect("usage: evalbin <in.bin> <out.i16>"),
            &std::env::args().nth(3).expect("usage: evalbin <in.bin> <out.i16>"),
        ),
        Some("datacheck") => datagen::datacheck(
            &std::env::args().nth(2).expect("usage: datacheck <file.bin>"),
        ),
        Some("datagen") => {
            // datagen <games> <out.bin> [seed] [threads]
            let args: Vec<String> = std::env::args().collect();
            let games = args.get(2).and_then(|n| n.parse().ok()).unwrap_or(1000);
            let out = args.get(3).cloned().unwrap_or_else(|| "data.bin".into());
            let seed = args.get(4).and_then(|n| n.parse().ok()).unwrap_or(0xDA7A);
            let threads = args.get(5).and_then(|n| n.parse().ok()).unwrap_or(4);
            datagen::datagen(games, &out, seed, threads);
        }
        Some("datagenseeded") => {
            // datagenseeded <games> <out.bin> <positions.epd> [seed] [threads]
            let args: Vec<String> = std::env::args().collect();
            let games = args.get(2).and_then(|n| n.parse().ok()).unwrap_or(1000);
            let out = args.get(3).cloned().unwrap_or_else(|| "data.bin".into());
            let positions = args.get(4).expect("usage: datagenseeded <games> <out.bin> <positions.epd> [seed] [threads]");
            let seed = args.get(5).and_then(|n| n.parse().ok()).unwrap_or(0xDA7A);
            let threads = args.get(6).and_then(|n| n.parse().ok()).unwrap_or(4);
            datagen::datagen_seeded(games, &out, seed, threads, positions);
        }
        _ => uci::uci_loop(),
    }
}

// ---------------- opening-book generation ----------------
// Random 8-ply walks from startpos, kept only if the engine itself judges the result balanced
// (|eval| <= 120cp at depth 8) and the game is ongoing. Fixed seed => reproducible book.

fn genbook(count: usize) {
    let mut seed: u64 = 0xC0FFEE + 1017;
    let mut rng = move || {
        // xorshift64
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut made = 0;
    let mut tried = 0;
    while made < count && tried < count * 50 {
        tried += 1;
        let mut board = Board::startpos();
        let mut ok = true;
        for _ in 0..8 {
            let mut moves = Vec::new();
            board.generate_moves(|pm| {
                for mv in pm {
                    moves.push(mv);
                }
                false
            });
            if moves.is_empty() {
                ok = false;
                break;
            }
            let mv = moves[(rng() % moves.len() as u64) as usize];
            board.play_unchecked(mv);
        }
        if !ok || board.status() != GameStatus::Ongoing {
            continue;
        }
        let mut s = Searcher::new(16);
        s.silent = true;
        let mut score_ok = false;
        // reuse think() for the balance filter via a tiny depth-limited probe
        s.think(
            &board,
            &Limits {
                depth: Some(8),
                nodes: Some(300_000),
                ..Default::default()
            },
        );
        // read the root score back from the TT
        if let Some(e) = s.tt.probe(board.hash()) {
            score_ok = (e.score as i32).abs() <= 120;
        }
        if score_ok {
            // EPD: piece placement, stm, castling, ep
            println!("{}", board);
            made += 1;
        }
    }
    eprintln!("generated {made} balanced openings ({tried} walks tried)");
}

// ---------------- gate: opening book (every line legality-walked) ----------------

fn book_gate() {
    println!("GATE book — every repertoire line legality-walked from startpos");
    match book::Book::load() {
        Ok(b) => {
            // spot checks: startpos offers e2e4 and ONLY e2e4 (KID lines must not
            // register 1.d4 for White); Locock, Ruy and the KID reply all reachable
            let start = Board::startpos();
            let mut seed = 1u64;
            let mut start_moves = std::collections::HashSet::new();
            for _ in 0..64 {
                if let Some(mv) = b.probe(start.hash(), &mut seed) {
                    start_moves.insert(mv.to_string());
                }
            }
            let walk = |toks: &[&str]| {
                let mut bd = Board::startpos();
                for t in toks {
                    let mv = cozy_chess::util::parse_uci_move(&bd, t).unwrap();
                    bd.play_unchecked(mv);
                }
                bd
            };
            let can_reach = |bd: &Board, want: &str, seed: &mut u64| {
                (0..64).any(|_| {
                    b.probe(bd.hash(), seed)
                        .map(|m| m.to_string() == want)
                        .unwrap_or(false)
                })
            };
            let locock = walk(&["e2e4", "e7e5", "g1f3", "d7d6", "d2d4", "g8f6"]);
            let ruy = walk(&["e2e4", "e7e5", "g1f3", "b8c6"]);
            let d4 = walk(&["d2d4"]);
            let has_locock = can_reach(&locock, "f3g5", &mut seed);
            let has_ruy = can_reach(&ruy, "f1b5", &mut seed);
            let has_kid = can_reach(&d4, "g8f6", &mut seed);
            let start_ok = start_moves.len() == 1 && start_moves.contains("e2e4");
            println!(
                "  {} positions | startpos moves: {:?} (must be exactly e2e4) | \
                 Locock: {} | Ruy 3.Bb5: {} | KID 1...Nf6: {}",
                b.positions(), start_moves, has_locock, has_ruy, has_kid
            );
            let ok = start_ok && has_locock && has_ruy && has_kid;
            println!("  => {}", if ok { "GATE PASS" } else { "GATE FAIL" });
            if !ok {
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("  {e}\n  => GATE FAIL");
            std::process::exit(1);
        }
    }
}

// ---------------- gate: NNUE int-exactness vs Python reference ----------------
// Rust quantized forward must equal nnue/ref_forward.py TO THE INTEGER on every record.
// Any mismatch = stop; a mostly-matching net is a silent-wrongness machine.

fn nnuegate(data_path: &str, net_path: &str, ref_path: &str) {
    println!("GATE nnue — Rust quantized forward vs Python integer reference, exact match");
    let net = nnue::load(net_path).expect("cannot load net");
    let data = std::fs::read(data_path).expect("cannot read dataset");
    let refs = std::fs::read(ref_path).expect("cannot read reference");
    let n = data.len() / datagen::RECORD_SIZE;
    assert_eq!(refs.len(), n * 4, "reference count mismatch");
    let mut mismatches = 0usize;
    let mut first: Option<(usize, i32, i32)> = None;
    for i in 0..n {
        let pos: [u8; 27] = data
            [i * datagen::RECORD_SIZE..i * datagen::RECORD_SIZE + 27]
            .try_into()
            .unwrap();
        let board = Board::from_fen(&datagen::unpack_to_fen(&pos), false).expect("unpack");
        let got = nnue::eval_scratch(&net, &board);
        let want = i32::from_le_bytes(refs[i * 4..i * 4 + 4].try_into().unwrap());
        if got != want {
            mismatches += 1;
            if first.is_none() {
                first = Some((i, got, want));
            }
        }
    }
    println!("  {n} positions | {mismatches} mismatches");
    if let Some((i, got, want)) = first {
        println!("  first mismatch: record {i} rust {got} vs ref {want}");
    }
    println!("  => {}", if mismatches == 0 { "GATE PASS" } else { "GATE FAIL" });
    if mismatches != 0 {
        std::process::exit(1);
    }
}

// ---------------- gate support: policy logit dump (verified by nnue/check_policy.py) ----------------

fn policydump(data_path: &str, out_path: &str) {
    use std::io::Write;
    let net = nnue::net().expect("set PATZER_EVALFILE");
    let pol = policy::policy().expect("set PATZER_POLICYFILE");
    let data = std::fs::read(data_path).expect("cannot read dataset");
    let n = (data.len() / datagen::RECORD_SIZE).min(2000);
    let mut out = std::io::BufWriter::new(std::fs::File::create(out_path).unwrap());
    for i in 0..n {
        let pos: [u8; 27] = data[i * datagen::RECORD_SIZE..i * datagen::RECORD_SIZE + 27]
            .try_into()
            .unwrap();
        let board = Board::from_fen(&datagen::unpack_to_fen(&pos), false).expect("unpack");
        let acc = nnue::acc_from(net, &board);
        let act = policy::activations(&acc, board.side_to_move());
        board.generate_moves(|pm| {
            for mv in pm {
                let c = policy::move_class(&board, mv);
                writeln!(out, "{i} {c} {}", policy::logit(pol, &act, c)).unwrap();
            }
            false
        });
    }
    println!("policydump: {n} positions -> {out_path}");
}

// ---------------- gate: incremental accumulator == from-scratch (perft for updates) ----------------

fn nnueinc(net_path: &str) {
    println!("GATE nnueinc — incremental accumulator vs from-scratch rebuild, every move of 1000 random games");
    let net = nnue::load(net_path).expect("cannot load net");
    let mut seed = 0xACC0_1234_u64;
    let mut rng = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut moves_checked = 0u64;
    let (mut castles, mut eps, mut promos) = (0u64, 0u64, 0u64);
    for _ in 0..1000 {
        let mut board = Board::startpos();
        let mut acc = nnue::acc_from(&net, &board);
        for _ in 0..200 {
            if board.status() != GameStatus::Ongoing {
                break;
            }
            let mut moves = Vec::new();
            board.generate_moves(|pm| {
                for mv in pm {
                    moves.push(mv);
                }
                false
            });
            let mv = moves[(rng() % moves.len() as u64) as usize];
            if board.color_on(mv.to) == Some(board.side_to_move()) {
                castles += 1;
            }
            if mv.promotion.is_some() {
                promos += 1;
            }
            if board.piece_on(mv.from) == Some(cozy_chess::Piece::Pawn)
                && mv.from.file() != mv.to.file()
                && board.piece_on(mv.to).is_none()
            {
                eps += 1;
            }
            acc = nnue::acc_update(&net, &acc, &board, mv);
            board.play_unchecked(mv);
            let scratch = nnue::acc_from(&net, &board);
            if acc.w != scratch.w || acc.b != scratch.b {
                println!("  MISMATCH after {mv} at\n  {board}");
                println!("  => GATE FAIL");
                std::process::exit(1);
            }
            moves_checked += 1;
        }
    }
    println!(
        "  {moves_checked} moves exact ({castles} castles, {eps} en-passants, {promos} promotions covered)"
    );
    println!("  => GATE PASS");
}

// ---------------- gate: SEE (hand-computable exchanges only — no lore) ----------------

fn see_gate() {
    println!("GATE see — expected values are pure arithmetic on trivial exchanges");
    // (name, fen, uci move, expected SEE in cp)
    let suite = [
        (
            "PxP undefended",
            "k7/8/8/3p4/4P3/8/8/K7 w - - 0 1",
            "e4d5",
            82, // win a pawn, no recapture
        ),
        (
            "PxP defended by P",
            "k7/8/4p3/3p4/4P3/8/8/K7 w - - 0 1",
            "e4d5",
            0, // pawn for pawn
        ),
        (
            "QxP defended by P",
            "k7/8/4p3/3p4/8/8/3Q4/K7 w - - 0 1",
            "d2d5",
            82 - 1025, // queen falls to the pawn recapture
        ),
    ];
    let mut all_ok = true;
    for (name, fen, mv_str, expected) in suite {
        let board = Board::from_fen(fen, false).expect("bad FEN");
        let mv = cozy_chess::util::parse_uci_move(&board, mv_str).expect("bad move");
        let got = search::see(&board, mv);
        let ok = got == expected;
        all_ok &= ok;
        println!(
            "  {:<20} {}  see {:>6} vs {:>6}  {}",
            name,
            mv_str,
            got,
            expected,
            if ok { "PASS" } else { "FAIL" }
        );
    }
    println!("  => {}", if all_ok { "GATE PASS" } else { "GATE FAIL" });
    if !all_ok {
        std::process::exit(1);
    }
}

// ---------------- gate 1: perft (from Stage 0) ----------------

fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    if depth == 1 {
        let mut n = 0u64;
        board.generate_moves(|moves| {
            n += moves.len() as u64;
            false
        });
        return n;
    }
    let mut nodes = 0u64;
    board.generate_moves(|moves| {
        for mv in moves {
            let mut b = board.clone();
            b.play_unchecked(mv);
            nodes += perft(&b, depth - 1);
        }
        false
    });
    nodes
}

const PERFT_SUITE: &[(&str, &str, &[(u32, u64)])] = &[
    (
        "startpos",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        &[(5, 4_865_609), (6, 119_060_324)],
    ),
    (
        "kiwipete",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        &[(4, 4_085_603), (5, 193_690_690)],
    ),
    (
        "pos3 (pins/ep)",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        &[(5, 674_624), (6, 11_030_083)],
    ),
    (
        "pos4 (promos)",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        &[(4, 422_333), (5, 15_833_292)],
    ),
    (
        "pos5",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        &[(4, 2_103_487), (5, 89_941_194)],
    ),
    (
        "pos6",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        &[(4, 3_894_594), (5, 164_075_551)],
    ),
];

fn perft_gate() {
    println!("GATE perft — exact node counts, any mismatch = FAIL");
    let mut all_ok = true;
    let t0 = std::time::Instant::now();
    let mut total = 0u64;
    for (name, fen, cases) in PERFT_SUITE {
        let board = Board::from_fen(fen, false).expect("bad FEN in suite");
        for &(depth, expected) in *cases {
            let got = perft(&board, depth);
            total += got;
            let ok = got == expected;
            all_ok &= ok;
            println!(
                "  {:<15} d{}  {:>13} vs {:>13}  {}",
                name,
                depth,
                got,
                expected,
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "  {} nodes, {:.2}s ({:.0} Mnps) => {}",
        total,
        secs,
        total as f64 / secs / 1e6,
        if all_ok { "GATE PASS" } else { "GATE FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
}

// ---------------- gate 2: self-verifying mate walker ----------------
// The harness does not trust the engine's score. It plays the engine's move and PROVES the mate
// by exhaustive walk over all replies, using only (perft-verified) movegen + status().

fn engine_move(board: &Board, depth: i32) -> Option<Move> {
    let mut s = Searcher::new(16);
    s.silent = true;
    s.think(
        board,
        &Limits {
            depth: Some(depth),
            nodes: Some(5_000_000),
            ..Default::default()
        },
    )
}

fn verify_mate(board: &Board, n: i32) -> bool {
    let mv = match engine_move(board, 2 * n + 2) {
        Some(mv) => mv,
        None => return false,
    };
    let mut b = board.clone();
    b.play_unchecked(mv);
    if b.status() == GameStatus::Won {
        return true; // opponent to move is checkmated — proven, not claimed
    }
    if n <= 1 || b.status() != GameStatus::Ongoing {
        return false;
    }
    // every opponent reply must be mated within n-1
    let mut replies = Vec::new();
    b.generate_moves(|pm| {
        for r in pm {
            replies.push(r);
        }
        false
    });
    replies.iter().all(|&r| {
        let mut b2 = b.clone();
        b2.play_unchecked(r);
        verify_mate(&b2, n - 1)
    })
}

fn mates_gate() {
    println!("GATE mates — engine's mate is PROVEN by exhaustive walk, not trusted");
    let suite: &[(&str, &str, i32)] = &[
        ("back-rank", "6k1/5ppp/8/8/8/8/8/4R2K w - - 0 1", 1),
        (
            // 1.e4 e5 2.Qh5 Nc6 3.Bc4 Nf6?? — Qh5xf7# via g6, NOT the Qf3 line (there ...Nf6
            // blocks the f-file; the first version of this suite had that wrong FEN and the
            // walker correctly failed it — the gate caught its author, not the engine).
            "scholar's",
            "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
            1,
        ),
        ("KQ corner", "7k/5K2/8/8/8/8/8/6Q1 w - - 0 1", 1),
        ("spite-block", "6k1/2r2ppp/8/8/8/8/8/1R5K w - - 0 1", 2),
        ("rook roller", "7k/8/8/8/8/8/R7/1R5K w - - 0 1", 2),
    ];
    let mut all_ok = true;
    for (name, fen, n) in suite {
        let board = Board::from_fen(fen, false).expect("bad FEN");
        let t0 = std::time::Instant::now();
        let ok = verify_mate(&board, *n);
        all_ok &= ok;
        println!(
            "  {:<12} mate-in-{}  {:>7.2?}  {}",
            name,
            n,
            t0.elapsed(),
            if ok { "PASS (proven)" } else { "FAIL" }
        );
    }
    println!("  => {}", if all_ok { "GATE PASS" } else { "GATE FAIL" });
    if !all_ok {
        std::process::exit(1);
    }
}

// ---------------- gate 3: bench (node-count signature) ----------------

fn bench() {
    const DEPTH: i32 = 11;
    println!("BENCH — fixed depth {DEPTH}, fresh TT per position; total nodes = functional signature");
    let mut total_nodes = 0u64;
    let t0 = std::time::Instant::now();
    for (name, fen, _) in PERFT_SUITE {
        let board = Board::from_fen(fen, false).expect("bad FEN");
        let mut s = Searcher::new(64);
        s.silent = true;
        let mv = s.think(
            &board,
            &Limits {
                depth: Some(DEPTH),
                ..Default::default()
            },
        );
        total_nodes += s.nodes;
        println!(
            "  {:<15} nodes {:>10}  best {}",
            name,
            s.nodes,
            mv.map(|m| cozy_chess::util::display_uci_move(&board, m).to_string())
                .unwrap_or_else(|| "-".into())
        );
    }
    let secs = t0.elapsed().as_secs_f64();
    println!(
        "  SIGNATURE: {} nodes | {:.2}s | {:.0} knps",
        total_nodes,
        secs,
        total_nodes as f64 / secs / 1e3
    );
}

// ---------------- gate 4: self-play soak ----------------

fn soak() {
    const GAMES: usize = 12;
    const NODES: u64 = 25_000;
    println!("SOAK — {GAMES} self-play games @ {NODES} nodes/move; every move legality-checked");
    let (mut w, mut d, mut l) = (0, 0, 0);
    for g in 0..GAMES {
        let mut board = Board::startpos();
        let mut s = Searcher::new(16);
        s.silent = true;
        let mut counts: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();
        let mut hist: Vec<u64> = Vec::new();
        let mut ply = 0;
        let result;
        loop {
            match board.status() {
                GameStatus::Won => {
                    // side to move is mated; the previous mover won
                    result = if ply % 2 == 1 { "1-0" } else { "0-1" };
                    break;
                }
                GameStatus::Drawn => {
                    result = "1/2";
                    break;
                }
                GameStatus::Ongoing => {}
            }
            let c = counts.entry(board.hash()).or_insert(0);
            *c += 1;
            if *c >= 3 || board.halfmove_clock() >= 100 || ply > 400 {
                result = "1/2";
                break;
            }
            s.game_hist = hist.clone();
            // vary the very first move a little across games via depth jitter
            let lim = Limits {
                nodes: Some(NODES + (g as u64 * 1017) % 5000),
                ..Default::default()
            };
            let mv = match s.think(&board, &lim) {
                Some(mv) => mv,
                None => {
                    println!("  game {g}: engine returned no move in ongoing position — FAIL");
                    std::process::exit(1);
                }
            };
            assert!(board.is_legal(mv), "ILLEGAL MOVE game {g} ply {ply}: {mv}");
            hist.push(board.hash());
            board.play_unchecked(mv);
            ply += 1;
        }
        match result {
            "1-0" => w += 1,
            "0-1" => l += 1,
            _ => d += 1,
        }
        println!("  game {:>2}: {:>3} plies, {}", g, ply, result);
    }
    println!("  => GATE PASS: {GAMES} games, 0 illegal moves, 0 panics (W{w} D{d} L{l})");
}
