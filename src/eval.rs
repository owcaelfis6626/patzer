//! Tapered evaluation: PeSTO material + piece-square tables (pesto.rs, machine-extracted).
//! Score is side-to-move relative (negamax convention).
//! Stage-1.75 HCE terms (mobility/pawns/king safety) tested NULL at equal time 2026-07-11
//! (1600 games, -2.8 +/- 14.0, cap-limited) — not merged; source kept in attic/.

use crate::pesto::*;
use cozy_chess::{Board, Color, Piece};

const PHASE_W: [i32; 6] = [0, 1, 1, 2, 4, 0]; // pawn..king, total start weight = 24

fn tables(p: Piece) -> (&'static [i32; 64], &'static [i32; 64]) {
    match p {
        Piece::Pawn => (&MG_PAWN_TABLE, &EG_PAWN_TABLE),
        Piece::Knight => (&MG_KNIGHT_TABLE, &EG_KNIGHT_TABLE),
        Piece::Bishop => (&MG_BISHOP_TABLE, &EG_BISHOP_TABLE),
        Piece::Rook => (&MG_ROOK_TABLE, &EG_ROOK_TABLE),
        Piece::Queen => (&MG_QUEEN_TABLE, &EG_QUEEN_TABLE),
        Piece::King => (&MG_KING_TABLE, &EG_KING_TABLE),
    }
}

pub fn evaluate(board: &Board) -> i32 {
    let mut mg = 0i32; // white minus black
    let mut eg = 0i32;
    let mut phase = 0i32;
    for sq in board.occupied() {
        let p = board.piece_on(sq).unwrap();
        let c = board.color_on(sq).unwrap();
        let (mgt, egt) = tables(p);
        let pi = p as usize;
        let si = sq as usize;
        let (idx, sign) = if c == Color::White { (si ^ 56, 1) } else { (si, -1) };
        mg += sign * (MG_VALUE[pi] + mgt[idx]);
        eg += sign * (EG_VALUE[pi] + egt[idx]);
        phase += PHASE_W[pi];
    }
    let phase = phase.min(24);
    let score = (mg * phase + eg * (24 - phase)) / 24;
    if board.side_to_move() == Color::White {
        score
    } else {
        -score
    }
}

/// piece value in centipawns (mg) for move ordering / delta pruning
pub fn piece_val(p: Piece) -> i32 {
    MG_VALUE[p as usize]
}
