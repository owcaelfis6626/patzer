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
            s.silent = i != 0;
            Arc::new(Mutex::new(s))
        })
        .collect();
    (tt, pool)
}

pub fn uci_loop() {
    let mut hash_mb: usize = 64;
    let mut n_threads: usize = 1;
    let stop = Arc::new(AtomicBool::new(false));
    let (mut tt, mut pool) = build_pool(hash_mb, n_threads, &stop);
    let mut board = Board::startpos();
    let mut game_hist: Vec<u64> = Vec::new();
    // this side's own root scores across the game, for volatility-aware time management
    let mut eval_hist_game: Vec<i32> = Vec::new();
    let book = crate::book::Book::load().expect("book failed legality walk");
    let mut own_book = true;
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
                println!("option name OwnBook type check default true");
                println!("option name EvalFile type string default <empty>");
                println!("option name PolicyFile type string default <empty>");
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
                    }
                }
            }
            Some("ucinewgame") => {
                tt.clear();
            }
            Some("position") => {
                let mut idx = 1;
                if tokens.get(idx) == Some(&"startpos") {
                    board = Board::startpos();
                    idx += 1;
                } else if tokens.get(idx) == Some(&"fen") {
                    let end = tokens
                        .iter()
                        .position(|&t| t == "moves")
                        .unwrap_or(tokens.len());
                    let fen = tokens[idx + 1..end].join(" ");
                    match Board::from_fen(&fen, false) {
                        Ok(b) => board = b,
                        Err(e) => {
                            println!("info string bad fen: {e}");
                            continue;
                        }
                    }
                    idx = end;
                }
                game_hist.clear();
                eval_hist_game.clear();
                if tokens.get(idx) == Some(&"moves") {
                    for mv_str in &tokens[idx + 1..] {
                        match parse_uci_move(&board, mv_str) {
                            Ok(mv) => {
                                game_hist.push(board.hash());
                                board.play_unchecked(mv);
                            }
                            Err(e) => {
                                println!("info string bad move {mv_str}: {e}");
                                break;
                            }
                        }
                    }
                }
            }
            Some("go") => {
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
                let mut it = tokens[1..].iter();
                while let Some(&tok) = it.next() {
                    let mut num =
                        |it: &mut std::slice::Iter<&str>| it.next().and_then(|v| v.parse().ok());
                    match tok {
                        "depth" => limits.depth = num(&mut it).map(|v: i64| v as i32),
                        "nodes" => limits.nodes = num(&mut it).map(|v: i64| v as u64),
                        "movetime" => limits.movetime = num(&mut it).map(|v: i64| v as u128),
                        "wtime" => limits.wtime = num(&mut it).map(|v: i64| v.max(1) as u128),
                        "btime" => limits.btime = num(&mut it).map(|v: i64| v.max(1) as u128),
                        "winc" => limits.winc = num(&mut it).map(|v: i64| v.max(0) as u128),
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
                if had_prev {
                    if let Ok(s0) = pool[0].lock() {
                        eval_hist_game.push(s0.last_score);
                    }
                }
                stop.store(false, Ordering::Relaxed); // reset once, before any thread starts
                for (i, s) in pool.iter().enumerate() {
                    let s = s.clone();
                    let board = board.clone();
                    let hist = game_hist.clone();
                    let evh = eval_hist_game.clone();
                    let limits = limits.clone();
                    let is_main = i == 0;
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
                            match best {
                                Some(mv) => println!("bestmove {}", display_uci_move(&board, mv)),
                                None => println!("bestmove 0000"),
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
