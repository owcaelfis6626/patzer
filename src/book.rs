//! Internal opening book: the boss's repertoire, registered position -> move along each
//! line, random choice where lines diverge. Every line is legality-walked by the `book`
//! gate before being trusted. Each line carries a side marker: 'w' registers White's
//! moves only, 'b' Black's only, '*' both — so e.g. the KID lines teach patzer to MEET
//! 1.d4 without making him OPEN 1.d4. The Locock move order (ECO C41) was verified
//! against 365chess, not memory — after the scholar's-FEN incident this codebase does
//! not encode obscure chess lore unverified; the Ruy Lopez / KID mainlines below are
//! core theory and are structure-checked by the gate's spot probes.

use cozy_chess::util::parse_uci_move;
use cozy_chess::{Board, Color, Move};
use std::collections::HashMap;

const LINES: &[(char, &str)] = &[
    // ---------------- Philidor personality (original) ----------------
    // Philidor Defence -> Nimzovich. (The Locock 5.Ng5 line was removed 2026-08-10:
    // a club in-joke, not a move worth playing. Objective opening choice now comes
    // from the result-weighted Polyglot book, not from this repertoire.)
    // deliberately NOT encoded: once out of book the engine searches.
    // Philidor, exchange line
    ('*', "e2e4 e7e5 g1f3 d7d6 d2d4 e5d4 f3d4 g8f6 b1c3 f8e7"),
    // Philidor, Nimzovich with 4.Nc3 Nbd7
    ('*', "e2e4 e7e5 g1f3 d7d6 d2d4 g8f6 b1c3 b8d7"),
    // Hyperaccelerated Dragon
    ('*', "e2e4 c7c5 g1f3 g7g6 d2d4 c5d4 f3d4 b8c6 b1c3 f8g7"),
    // ---------------- Ruy Lopez (White repertoire, 2026-07-14) ----------------
    // Closed mainline: ...d6 order, 9.h3
    ('*', "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 d7d6 c2c3 e8g8 h2h3"),
    // Closed, castles-first order -> 8.a4 anti-Marshall
    ('*', "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f8e7 f1e1 b7b5 a4b3 e8g8 a2a4"),
    // Berlin endgame
    ('*', "e2e4 e7e5 g1f3 b8c6 f1b5 g8f6 e1g1 f6e4 d2d4 e4d6 b5c6 d7c6 d4e5 d6f5 d1d8 e8d8"),
    // Open mainline
    ('*', "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 f6e4 d2d4 b7b5 a4b3 d7d5 d4e5 c8e6"),
    // Steinitz deferred
    ('*', "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 d7d6 c2c3"),
    // Neo-Arkhangelsk
    ('*', "e2e4 e7e5 g1f3 b8c6 f1b5 a7a6 b5a4 g8f6 e1g1 b7b5 a4b3 f8c5 c2c3"),
    // Schliemann: principled 4.Nc3
    ('*', "e2e4 e7e5 g1f3 b8c6 f1b5 f7f5 b1c3"),
    // Bird's Defence
    ('*', "e2e4 e7e5 g1f3 b8c6 f1b5 c6d4 f3d4 e5d4 e1g1"),
    // Classical (3...Bc5): 4.c3
    ('*', "e2e4 e7e5 g1f3 b8c6 f1b5 f8c5 c2c3 g8f6 d2d4"),
    // ---------------- King's Indian (Black repertoire vs 1.d4, 2026-07-14) ----------
    // Classical Mar del Plata to 9...Ne7
    ('b', "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 g1f3 e8g8 f1e2 e7e5 e1g1 b8c6 d4d5 c6e7"),
    // Saemisch: 6...e5
    ('b', "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f3 e8g8 c1e3 e7e5"),
    // Four Pawns Attack: ...c5
    ('b', "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f2f4 e8g8 g1f3 c7c5"),
    // Averbakh: ...Na6
    ('b', "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7 e2e4 d7d6 f1e2 e8g8 c1g5 b8a6"),
    // Fianchetto (Nf3+g3 order): ...Nbd7
    ('b', "d2d4 g8f6 c2c4 g7g6 g1f3 f8g7 g2g3 e8g8 f1g2 d7d6 e1g1 b8d7"),
    // Fianchetto (g3-first order) — transposes by hash
    ('b', "d2d4 g8f6 c2c4 g7g6 g2g3 f8g7 f1g2 e8g8 b1c3 d7d6 g1f3 b8d7"),
    // London vs KID setup
    ('b', "d2d4 g8f6 g1f3 g7g6 c1f4 f8g7 e2e3 d7d6"),
    // Trompowsky: 2...Ne4
    ('b', "d2d4 g8f6 c1g5 f6e4 g5f4 d7d5"),
];

pub struct Book {
    map: HashMap<u64, Vec<Move>>,
}

impl Book {
    /// Walk every line from startpos, validating each move; register a move only if the
    /// line's side marker covers the mover. Any illegal/unparsable move is a hard error
    /// naming the line — the gate runs this and fails loudly.
    pub fn load() -> Result<Book, String> {
        let mut map: HashMap<u64, Vec<Move>> = HashMap::new();
        for (side, line) in LINES {
            let mut board = Board::startpos();
            for tok in line.split_whitespace() {
                let mv = parse_uci_move(&board, tok)
                    .map_err(|e| format!("book line '{line}': move {tok}: {e}"))?;
                let mover_covered = match side {
                    '*' => true,
                    'w' => board.side_to_move() == Color::White,
                    'b' => board.side_to_move() == Color::Black,
                    _ => return Err(format!("bad side marker '{side}'")),
                };
                if mover_covered {
                    let entry = map.entry(board.hash()).or_default();
                    if !entry.contains(&mv) {
                        entry.push(mv);
                    }
                }
                board.play_unchecked(mv);
            }
        }
        Ok(Book { map })
    }

    pub fn positions(&self) -> usize {
        self.map.len()
    }

    pub fn probe(&self, hash: u64, seed: &mut u64) -> Option<Move> {
        let choices = self.map.get(&hash)?;
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        Some(choices[(*seed % choices.len() as u64) as usize])
    }
}
