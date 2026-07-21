//! Stage 3 — policy head riding the shared (frozen v2) accumulator. File "PZP1":
//! magic | u32 n_classes (24576 = piece*4096 + from*64 + to) | u32 dim (512) |
//! i8 W[n_classes][dim]. The head scores QUIET moves for ordering only (arm A):
//! logit(move) = <act512, W[class]> where act512 = clamp(acc.us ++ acc.them, 0, 127),
//! exactly the integers the value head consumes. Ordering is scale-free, so the i8
//! quantization is a monotone transform of the trained logits.

use crate::nnue::{Acc, ACC};
use cozy_chess::{Board, Color, Move};
use std::sync::OnceLock;

pub const N_CLASSES: usize = 6 * 64 * 64;
pub const DIM: usize = 2 * ACC;

pub struct Policy {
    pub w: Vec<i8>, // [N_CLASSES][DIM]
}

static POLICY: OnceLock<Policy> = OnceLock::new();

pub fn load_global(path: &str) -> Result<(), String> {
    let p = load(path)?;
    POLICY.set(p).map_err(|_| "PolicyFile already loaded".to_string())?;
    Ok(())
}

pub fn policy() -> Option<&'static Policy> {
    POLICY.get()
}

pub fn load(path: &str) -> Result<Policy, String> {
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if data.len() < 12 || &data[0..4] != b"PZP1" {
        return Err(format!("{path}: bad magic"));
    }
    let nc = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    if nc != N_CLASSES || dim != DIM {
        return Err(format!("{path}: shape {nc}x{dim}, engine wants {N_CLASSES}x{DIM}"));
    }
    if data.len() != 12 + nc * dim {
        return Err(format!("{path}: bad size"));
    }
    let w = data[12..].iter().map(|&b| b as i8).collect();
    Ok(Policy { w })
}

/// act512 in 0..127 from the perspective pair (stm half first, matching the value head)
#[inline]
pub fn activations(acc: &Acc, stm: Color) -> [i32; DIM] {
    let (us, them) = match stm {
        Color::White => (&acc.w, &acc.b),
        Color::Black => (&acc.b, &acc.w),
    };
    let mut act = [0i32; DIM];
    for i in 0..ACC {
        act[i] = (us[i] as i32).clamp(0, 127);
        act[ACC + i] = (them[i] as i32).clamp(0, 127);
    }
    act
}

/// class index must match train_policy.py: mover_piece*4096 + from*64 + to
#[inline]
pub fn move_class(board: &Board, mv: Move) -> usize {
    let piece = board.piece_on(mv.from).expect("no piece on from") as usize;
    piece * 4096 + (mv.from as usize) * 64 + (mv.to as usize)
}

#[inline]
pub fn logit(pol: &Policy, act: &[i32; DIM], class: usize) -> i32 {
    let row = &pol.w[class * DIM..(class + 1) * DIM];
    let mut s = 0i32;
    for (a, &w) in act.iter().zip(row) {
        s += a * w as i32;
    }
    s
}
