//! Stage 2c — quantized NNUE inference (768-basic: 2×256 perspective accumulator → CReLU →
//! 16 → CReLU → 1). File format "PZR1" is produced by nnue/train.py; all integer arithmetic
//! here must match nnue/ref_forward.py TO THE INTEGER (gate: `patzer nnuegate`), because a
//! mostly-matching net is a silent-wrongness machine.
//!
//! Scales: ft weights/bias i16 at 127/1.0; l1 w i8 at 64, b i32 at 127*64; l2 same.
//! Forward: acc(i16, clamp 0..127) -> l1 i32 -> clamp 0..8128 -> /64 (0..127) -> l2 i32 = v2;
//! cp = v2 * 400 / 8128, truncated toward zero (Rust i32 division semantics).

use cozy_chess::{Board, Color, Move};
use std::sync::OnceLock;

pub const ACC: usize = 256;
const HIDDEN: usize = 16;
const FT_Q: i32 = 127;
const W_Q: i32 = 64;
const K_CP: i32 = 400;

pub struct Network {
    pub ft: Vec<i16>,      // [input_dim][ACC]
    pub ft_bias: Vec<i16>, // [ACC]
    pub w1: Vec<i16>,      // [HIDDEN][2*ACC] — stored i8 on disk, widened at load for vpmaddwd
    pub b1: Vec<i32>,      // [HIDDEN]
    pub w2: Vec<i32>,      // [HIDDEN] — widened at load
    pub b2: i32,
    pub input_dim: usize,  // 768 = basic (PZR1), 40960 = HalfKP (PZRH)
}

impl Network {
    #[inline]
    pub fn is_halfkp(&self) -> bool {
        self.input_dim == 40960
    }
}

static NET: OnceLock<Network> = OnceLock::new();

pub fn load_global(path: &str) -> Result<(), String> {
    let net = load(path)?;
    NET.set(net).map_err(|_| "EvalFile already loaded".to_string())?;
    Ok(())
}

pub fn net() -> Option<&'static Network> {
    NET.get()
}

pub fn load(path: &str) -> Result<Network, String> {
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if data.len() < 12 {
        return Err(format!("{path}: too short"));
    }
    let (input_dim, header) = match &data[0..4] {
        b"PZR1" => (768usize, 8usize),
        b"PZRH" => (
            u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize,
            12usize,
        ),
        _ => return Err(format!("{path}: bad magic")),
    };
    let acc = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    if acc != ACC {
        return Err(format!("{path}: acc width {acc}, engine built for {ACC}"));
    }
    let expect =
        header + input_dim * ACC * 2 + ACC * 2 + HIDDEN * 2 * ACC + HIDDEN * 4 + HIDDEN + 4;
    if data.len() != expect {
        return Err(format!("{path}: size {} != expected {expect}", data.len()));
    }
    let mut off = header;
    let mut take_i16 = |n: usize| {
        let v: Vec<i16> = data[off..off + 2 * n]
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();
        off += 2 * n;
        v
    };
    let ft = take_i16(input_dim * ACC);
    let ft_bias = take_i16(ACC);
    let w1: Vec<i16> = data[off..off + HIDDEN * 2 * ACC]
        .iter()
        .map(|&b| b as i8 as i16)
        .collect();
    let mut off = off + HIDDEN * 2 * ACC;
    let b1: Vec<i32> = data[off..off + HIDDEN * 4]
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    off += HIDDEN * 4;
    let w2: Vec<i32> = data[off..off + HIDDEN]
        .iter()
        .map(|&b| b as i8 as i32)
        .collect();
    off += HIDDEN;
    let b2 = i32::from_le_bytes(data[off..off + 4].try_into().unwrap());
    Ok(Network { ft, ft_bias, w1, b1, w2, b2, input_dim })
}

/// feature index for one piece from one perspective; must match train.py to_features:
/// white view: color*384 + piece*64 + sq ; black view: (1-color)*384 + piece*64 + (sq^56)
#[inline]
fn feature(persp: Color, color: Color, piece: usize, sq: usize) -> usize {
    match persp {
        Color::White => (color as usize) * 384 + piece * 64 + sq,
        Color::Black => (1 - color as usize) * 384 + piece * 64 + (sq ^ 56),
    }
}

/// HalfKP feature; must match train.py to_features_halfkp. `ksq` is the perspective
/// owner's king square in BOARD coordinates (mirroring applied here). Kings are never
/// features (piece must be 0..4 = P,N,B,R,Q).
#[inline]
fn feature_hkp(persp: Color, ksq: usize, color: Color, piece: usize, sq: usize) -> usize {
    match persp {
        Color::White => ksq * 640 + (color as usize * 5 + piece) * 64 + sq,
        Color::Black => {
            (ksq ^ 56) * 640 + ((1 - color as usize) * 5 + piece) * 64 + (sq ^ 56)
        }
    }
}

/// from-scratch accumulator for one perspective
pub fn build_acc(net: &Network, board: &Board, persp: Color) -> [i16; ACC] {
    let mut acc = [0i16; ACC];
    acc.copy_from_slice(&net.ft_bias);
    let hkp = net.is_halfkp();
    let ksq = board.king(persp) as usize;
    for sq in board.occupied() {
        let p = board.piece_on(sq).unwrap() as usize;
        let c = board.color_on(sq).unwrap();
        let f = if hkp {
            if p == 5 {
                continue; // kings are buckets, not features
            }
            feature_hkp(persp, ksq, c, p, sq as usize)
        } else {
            feature(persp, c, p, sq as usize)
        };
        let row = &net.ft[f * ACC..(f + 1) * ACC];
        for (a, &w) in acc.iter_mut().zip(row) {
            *a += w;
        }
    }
    acc
}

/// quantized forward from the two accumulators; stm picks which half goes first
pub fn forward(net: &Network, acc_w: &[i16; ACC], acc_b: &[i16; ACC], stm: Color) -> i32 {
    let (us, them) = match stm {
        Color::White => (acc_w, acc_b),
        Color::Black => (acc_b, acc_w),
    };
    // i16 activations + i16 weights with i32 accumulation: the vpmaddwd shape.
    // Widening is value-exact, so this matches the Python reference to the integer.
    let mut act = [0i16; 2 * ACC];
    for i in 0..ACC {
        act[i] = us[i].clamp(0, FT_Q as i16);
        act[ACC + i] = them[i].clamp(0, FT_Q as i16);
    }
    let mut h = [0i32; HIDDEN];
    for (j, hj) in h.iter_mut().enumerate() {
        let row = &net.w1[j * 2 * ACC..(j + 1) * 2 * ACC];
        let mut s = 0i32;
        for (&a, &w) in act.iter().zip(row) {
            s += a as i32 * w as i32;
        }
        *hj = (s + net.b1[j]).clamp(0, FT_Q * W_Q) / W_Q; // requantize to 0..127
    }
    let mut v2 = net.b2;
    for (hj, &w) in h.iter().zip(&net.w2) {
        v2 += hj * w;
    }
    v2 * K_CP / (FT_Q * W_Q) // truncating division = the documented semantics
}

/// full evaluation from scratch (correctness path; incremental comes in 2c-ii)
pub fn eval_scratch(net: &Network, board: &Board) -> i32 {
    let acc_w = build_acc(net, board, Color::White);
    let acc_b = build_acc(net, board, Color::Black);
    forward(net, &acc_w, &acc_b, board.side_to_move())
}

// ---------------- incremental accumulator (2c-ii) ----------------
// For 768-basic every move is a handful of add/sub rows — no rebuilds ever. The update is
// computed from (board, mv) BEFORE the move is played; cozy castling is king-takes-rook.
// Correctness gate: `patzer nnueinc` (incremental == scratch after every move of random games).

#[derive(Clone)]
pub struct Acc {
    pub w: [i16; ACC],
    pub b: [i16; ACC],
}

pub fn acc_from(net: &Network, board: &Board) -> Acc {
    Acc {
        w: build_acc(net, board, Color::White),
        b: build_acc(net, board, Color::Black),
    }
}

#[inline]
fn row_add(acc: &mut Acc, net: &Network, color: Color, piece: usize, sq: usize) {
    let fw = feature(Color::White, color, piece, sq);
    let fb = feature(Color::Black, color, piece, sq);
    let rw = &net.ft[fw * ACC..(fw + 1) * ACC];
    let rb = &net.ft[fb * ACC..(fb + 1) * ACC];
    for i in 0..ACC {
        acc.w[i] += rw[i];
        acc.b[i] += rb[i];
    }
}

#[inline]
fn row_sub(acc: &mut Acc, net: &Network, color: Color, piece: usize, sq: usize) {
    let fw = feature(Color::White, color, piece, sq);
    let fb = feature(Color::Black, color, piece, sq);
    let rw = &net.ft[fw * ACC..(fw + 1) * ACC];
    let rb = &net.ft[fb * ACC..(fb + 1) * ACC];
    for i in 0..ACC {
        acc.w[i] -= rw[i];
        acc.b[i] -= rb[i];
    }
}

/// accumulator after `mv` is played on `board` (board must be the pre-move position)
pub fn acc_update(net: &Network, acc: &Acc, board: &Board, mv: Move) -> Acc {
    if net.is_halfkp() {
        acc_update_hkp(net, acc, board, mv)
    } else {
        acc_update_basic(net, acc, board, mv)
    }
}

// ---- HalfKP incremental helpers: single-perspective row ops with explicit bucket ----

#[inline]
fn hkp_add(half: &mut [i16; ACC], net: &Network, persp: Color, ksq: usize,
           color: Color, piece: usize, sq: usize) {
    let f = feature_hkp(persp, ksq, color, piece, sq);
    let row = &net.ft[f * ACC..(f + 1) * ACC];
    for (a, &w) in half.iter_mut().zip(row) {
        *a += w;
    }
}

#[inline]
fn hkp_sub(half: &mut [i16; ACC], net: &Network, persp: Color, ksq: usize,
           color: Color, piece: usize, sq: usize) {
    let f = feature_hkp(persp, ksq, color, piece, sq);
    let row = &net.ft[f * ACC..(f + 1) * ACC];
    for (a, &w) in half.iter_mut().zip(row) {
        *a -= w;
    }
}

/// HalfKP update: an own-king move (incl. castling) rebuilds the mover's perspective
/// (its bucket changed); the opponent's perspective never rebuilds on a king move
/// (kings are not features) and takes only the material delta.
fn acc_update_hkp(net: &Network, acc: &Acc, board: &Board, mv: Move) -> Acc {
    use cozy_chess::{File, Piece, Square};
    let mut a = acc.clone();
    let stm = board.side_to_move();
    let them = match stm {
        Color::White => Color::Black,
        Color::Black => Color::White,
    };
    let moving = board.piece_on(mv.from).expect("no piece on from");
    let castling = board.color_on(mv.to) == Some(stm);

    if moving == Piece::King || castling {
        let mut nb = board.clone();
        nb.play_unchecked(mv);
        match stm {
            Color::White => a.w = build_acc(net, &nb, Color::White),
            Color::Black => a.b = build_acc(net, &nb, Color::Black),
        }
        let oksq = board.king(them) as usize;
        let other = match stm {
            Color::White => &mut a.b,
            Color::Black => &mut a.w,
        };
        if castling {
            // opponent's view: our rook relocates (king is not a feature)
            let back = mv.from.rank();
            let rf = if mv.to.file() > mv.from.file() { File::F } else { File::D };
            hkp_sub(other, net, them, oksq, stm, Piece::Rook as usize, mv.to as usize);
            hkp_add(other, net, them, oksq, stm, Piece::Rook as usize,
                    Square::new(rf, back) as usize);
        } else if let Some(victim) = board.piece_on(mv.to) {
            // king captured: opponent loses the victim (their own piece, their view)
            hkp_sub(other, net, them, oksq, them, victim as usize, mv.to as usize);
        }
        return a;
    }

    // non-king move: buckets unchanged for both perspectives — pure add/sub
    let wk = board.king(Color::White) as usize;
    let bk = board.king(Color::Black) as usize;
    let mut both_sub = |color: Color, piece: usize, sq: usize, a: &mut Acc| {
        hkp_sub(&mut a.w, net, Color::White, wk, color, piece, sq);
        hkp_sub(&mut a.b, net, Color::Black, bk, color, piece, sq);
    };
    let mut both_add = |color: Color, piece: usize, sq: usize, a: &mut Acc| {
        hkp_add(&mut a.w, net, Color::White, wk, color, piece, sq);
        hkp_add(&mut a.b, net, Color::Black, bk, color, piece, sq);
    };
    both_sub(stm, moving as usize, mv.from as usize, &mut a);
    if let Some(victim) = board.piece_on(mv.to) {
        both_sub(them, victim as usize, mv.to as usize, &mut a);
    } else if moving == Piece::Pawn && mv.from.file() != mv.to.file() {
        let vsq = Square::new(mv.to.file(), mv.from.rank());
        both_sub(them, Piece::Pawn as usize, vsq as usize, &mut a);
    }
    let placed = mv.promotion.map(|p| p as usize).unwrap_or(moving as usize);
    both_add(stm, placed, mv.to as usize, &mut a);
    a
}

fn acc_update_basic(net: &Network, acc: &Acc, board: &Board, mv: Move) -> Acc {
    use cozy_chess::{File, Piece, Square};
    let mut a = acc.clone();
    let stm = board.side_to_move();
    let them = match stm {
        Color::White => Color::Black,
        Color::Black => Color::White,
    };
    let moving = board.piece_on(mv.from).expect("no piece on from") as usize;

    if board.color_on(mv.to) == Some(stm) {
        // castling (cozy: king takes own rook)
        let back = mv.from.rank();
        let (kf, rf) = if mv.to.file() > mv.from.file() {
            (File::G, File::F)
        } else {
            (File::C, File::D)
        };
        row_sub(&mut a, net, stm, Piece::King as usize, mv.from as usize);
        row_sub(&mut a, net, stm, Piece::Rook as usize, mv.to as usize);
        row_add(&mut a, net, stm, Piece::King as usize, Square::new(kf, back) as usize);
        row_add(&mut a, net, stm, Piece::Rook as usize, Square::new(rf, back) as usize);
    } else {
        row_sub(&mut a, net, stm, moving, mv.from as usize);
        if let Some(victim) = board.piece_on(mv.to) {
            row_sub(&mut a, net, them, victim as usize, mv.to as usize);
        } else if moving == Piece::Pawn as usize && mv.from.file() != mv.to.file() {
            // en passant: captured pawn sits on (to.file, from.rank)
            let vsq = Square::new(mv.to.file(), mv.from.rank());
            row_sub(&mut a, net, them, Piece::Pawn as usize, vsq as usize);
        }
        let placed = mv.promotion.map(|p| p as usize).unwrap_or(moving);
        row_add(&mut a, net, stm, placed, mv.to as usize);
    }
    a
}
