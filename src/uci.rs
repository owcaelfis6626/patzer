//! Minimal-but-correct UCI. Search runs on a worker thread so `stop` is honored mid-search.
//! Castling conversion (cozy-chess king-takes-rook <-> standard UCI) is delegated to
//! cozy_chess::util::{parse_uci_move, display_uci_move}.

use crate::search::{Limits, Searcher};
use cozy_chess::util::{display_uci_move, parse_uci_move};
use cozy_chess::Board;
use std::io::BufRead;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};

pub fn uci_loop() {
    let searcher = Arc::new(Mutex::new(Searcher::new(64)));
    let stop = searcher.lock().unwrap().stop_flag();
    let mut board = Board::startpos();
    let mut game_hist: Vec<u64> = Vec::new();
    let book = crate::book::Book::load().expect("book failed legality walk");
    let mut own_book = true;
    let mut book_seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xC0FFEE)
        | 1;

    let mut worker: Option<std::thread::JoinHandle<()>> = None;
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
                            *searcher.lock().unwrap() = Searcher::new(mb);
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
                searcher.lock().unwrap().tt.clear();
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
                if let Some(h) = worker.take() {
                    let _ = h.join(); // previous search must have printed its bestmove
                }
                let searcher = searcher.clone();
                let board = board.clone();
                let hist = game_hist.clone();
                worker = Some(std::thread::spawn(move || {
                    let mut s = match searcher.try_lock() {
                        Ok(s) => s,
                        Err(_) => {
                            println!("info string already searching");
                            return;
                        }
                    };
                    s.game_hist = hist;
                    let best = s.think(&board, &limits);
                    match best {
                        Some(mv) => println!("bestmove {}", display_uci_move(&board, mv)),
                        None => println!("bestmove 0000"),
                    }
                }));
            }
            Some("stop") => stop.store(true, Ordering::Relaxed),
            Some("quit") => break,
            _ => {}
        }
    }
    stop.store(true, Ordering::Relaxed);
    if let Some(h) = worker.take() {
        let _ = h.join(); // let the search print bestmove before the process exits
    }
}
