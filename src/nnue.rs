//! Stage 2c — quantized NNUE inference (768-basic: 2×256 perspective accumulator → CReLU →
//! 16 → CReLU → 1). File format "PZR1" is produced by nnue/train.py; all integer arithmetic
//! here must match nnue/ref_forward.py TO THE INTEGER (gate: `patzer nnuegate`), because a
//! mostly-matching net is a silent-wrongness machine.
//!
//! Scales: ft weights/bias i16 at 127/1.0; l1 w i8 at 64, b i32 at 127*64; l2 same.
//! Forward: acc(i16, clamp 0..127) -> l1 i32 -> clamp 0..8128 -> /64 (0..127) -> l2 i32 = v2;
//! cp = v2 * 400 / 8128, truncated toward zero (Rust i32 division semantics).

use cozy_chess::{Board, Color, Move, Piece};
use std::sync::OnceLock;

pub const ACC: usize = 256;
const HIDDEN: usize = 16;
const FT_Q: i32 = 127;
const W_Q: i32 = 64;
const K_CP: i32 = 400;

pub struct Network {
    pub ft: Vec<i16>,      // [input_dim][ACC]
    pub ft_bias: Vec<i16>, // [ACC]
    pub w1: Vec<i8>,       // [HIDDEN][2*ACC] — i8 on disk AND in memory (see forward_avx2)
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
    let w1: Vec<i8> = data[off..off + HIDDEN * 2 * ACC]
        .iter()
        .map(|&b| b as i8)
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

/// quantized forward from the two accumulators; stm picks which half goes first.
///
/// Dispatch is COMPILE-TIME, not runtime: `.cargo/config.toml` pins `target-cpu=native` and
/// these binaries are machine-local, so there is nothing to detect. `forward_scalar` stays
/// compiled on every target -- it is the reference the Python gate (`patzer nnuegate`) is
/// written against, and `nnue::avx2_tests` holds the two together.
#[inline]
pub fn forward(net: &Network, acc_w: &[i16; ACC], acc_b: &[i16; ACC], stm: Color) -> i32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        // SAFETY: the cfg above is exactly the AVX2 precondition of forward_avx2.
        unsafe { forward_avx2(net, acc_w, acc_b, stm) }
    }
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
    {
        forward_scalar(net, acc_w, acc_b, stm)
    }
}

/// Reference forward. Matches nnue/ref_forward.py to the integer.
// Not called in an AVX2 build -- it is the reference the gate test compares against, and the
// fallback on any target without AVX2.
#[allow(dead_code)]
pub fn forward_scalar(net: &Network, acc_w: &[i16; ACC], acc_b: &[i16; ACC], stm: Color) -> i32 {
    let (us, them) = match stm {
        Color::White => (acc_w, acc_b),
        Color::Black => (acc_b, acc_w),
    };
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

/// AVX2 forward. 2026-09-09. Same integers as `forward_scalar`, ~3x faster.
///
/// WHAT WAS WRONG. The scalar loop is the "vpmaddwd shape" the old comment claimed, and LLVM
/// does emit vpmaddwd for it -- but at 128 bits, not 256. The vectorizer sizes the vector by
/// the i32 REDUCTION accumulator (8 lanes = one ymm), so the i16 inputs only ever fill an xmm:
/// 8 multiply-accumulates per instruction where the ISA offers 16. It cannot see that vpmaddwd
/// folds pairs of i16 down into i32, so a full ymm of inputs would still yield one ymm of
/// sums. Measured in the shipped binary: 4 static vpmaddwd, every one of them `%xmm`.
///
/// WHAT THIS DOES INSTEAD. Activations clamp to 0..127, so they fit u8; the weights are i8 on
/// disk. That is exactly vpmaddubsw's (u8 x i8) shape, which consumes 32 lanes per
/// instruction -- 4x the input width the compiler chose -- and halves the weight-matrix
/// traffic as a side effect, since w1 no longer has to be widened to i16 at load.
///
/// WHY IT IS BIT-EXACT, not merely close. vpmaddubsw saturates its i16 output, so the only
/// way to differ from the scalar sum is to reach that saturation. It cannot: |a*w| <= 127*128
/// = 16256, and the instruction adds two such products, so |result| <= 32512 < 32767. The
/// vpmaddwd-by-ones widening and the i32 tree-sum are exact, and i32 addition is associative,
/// so the reassociated total IS the scalar total. The bench signature is therefore unchanged,
/// which is the gate this change is held to: a pure-speed change that alters no tree.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn forward_avx2(net: &Network, acc_w: &[i16; ACC], acc_b: &[i16; ACC], stm: Color) -> i32 {
    use std::arch::x86_64::*;
    unsafe {
        let (us, them) = match stm {
            Color::White => (acc_w, acc_b),
            Color::Black => (acc_b, acc_w),
        };
        // clamp to 0..127 in i16, then pack the two halves down to u8 in the scalar layout
        // (`us` first, `them` second).
        let mut act = [0u8; 2 * ACC];
        let zero = _mm256_setzero_si256();
        let cap = _mm256_set1_epi16(FT_Q as i16);
        for (half, src) in [us.as_ptr(), them.as_ptr()].into_iter().enumerate() {
            for i in (0..ACC).step_by(32) {
                let a = _mm256_min_epi16(
                    _mm256_max_epi16(_mm256_loadu_si256(src.add(i) as *const __m256i), zero),
                    cap,
                );
                let b = _mm256_min_epi16(
                    _mm256_max_epi16(_mm256_loadu_si256(src.add(i + 16) as *const __m256i), zero),
                    cap,
                );
                // packus interleaves per 128-bit lane; permute4x64 puts the halves back in order
                let packed = _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0b11_01_10_00);
                _mm256_storeu_si256(act.as_mut_ptr().add(half * ACC + i) as *mut __m256i, packed);
            }
        }
        let ones = _mm256_set1_epi16(1);
        let mut h = [0i32; HIDDEN];
        for (j, hj) in h.iter_mut().enumerate() {
            let row = net.w1.as_ptr().add(j * 2 * ACC);
            // two accumulators: the maddubs->madd->paddd chain has latency, and a single chain
            // would serialise the loop on it.
            let mut s0 = _mm256_setzero_si256();
            let mut s1 = _mm256_setzero_si256();
            for i in (0..2 * ACC).step_by(64) {
                let a0 = _mm256_loadu_si256(act.as_ptr().add(i) as *const __m256i);
                let w0 = _mm256_loadu_si256(row.add(i) as *const __m256i);
                let a1 = _mm256_loadu_si256(act.as_ptr().add(i + 32) as *const __m256i);
                let w1 = _mm256_loadu_si256(row.add(i + 32) as *const __m256i);
                s0 = _mm256_add_epi32(s0, _mm256_madd_epi16(_mm256_maddubs_epi16(a0, w0), ones));
                s1 = _mm256_add_epi32(s1, _mm256_madd_epi16(_mm256_maddubs_epi16(a1, w1), ones));
            }
            let v = _mm256_add_epi32(s0, s1);
            let q = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
            let q = _mm_add_epi32(q, _mm_shuffle_epi32(q, 0b01_00_11_10));
            let q = _mm_add_epi32(q, _mm_shuffle_epi32(q, 0b00_01_00_01));
            *hj = (_mm_cvtsi128_si32(q) + net.b1[j]).clamp(0, FT_Q * W_Q) / W_Q;
        }
        let mut v2 = net.b2;
        for (hj, &w) in h.iter().zip(&net.w2) {
            v2 += hj * w;
        }
        v2 * K_CP / (FT_Q * W_Q)
    }
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
    use cozy_chess::Square;
    let moving = board.piece_on(mv.from).expect("no piece on from");
    // Cold callers (nnueinc, datagen) do not carry the victim; resolve it here. The search
    // passes it in, which is what removes this `piece_on` scan from the per-make path.
    let victim = match board.piece_on(mv.to) {
        Some(v) => Some((v, mv.to)),
        None if moving == Piece::Pawn && mv.from.file() != mv.to.file() => {
            Some((Piece::Pawn, Square::new(mv.to.file(), mv.from.rank())))
        }
        None => None,
    };
    acc_update_known(net, acc, board, mv, moving, victim)
}

/// As `acc_update`, but told which piece is moving and what it captures.
///
/// `victim` is the captured piece and the square it stood on -- for en passant that is
/// `(to.file, from.rank)`, NOT `mv.to`. Passing it in removes the `board.piece_on(mv.to)` scan
/// (up to six bitboard tests) that used to run on every make.
pub fn acc_update_known(
    net: &Network,
    acc: &Acc,
    board: &Board,
    mv: Move,
    moving: Piece,
    victim: Option<(Piece, cozy_chess::Square)>,
) -> Acc {
    if net.is_halfkp() {
        acc_update_hkp(net, acc, board, mv)
    } else {
        acc_update_basic(net, acc, board, mv, moving, victim)
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
    let both_sub = |color: Color, piece: usize, sq: usize, a: &mut Acc| {
        hkp_sub(&mut a.w, net, Color::White, wk, color, piece, sq);
        hkp_sub(&mut a.b, net, Color::Black, bk, color, piece, sq);
    };
    let both_add = |color: Color, piece: usize, sq: usize, a: &mut Acc| {
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

fn acc_update_basic(
    net: &Network,
    acc: &Acc,
    board: &Board,
    mv: Move,
    moving: Piece,
    victim: Option<(Piece, cozy_chess::Square)>,
) -> Acc {
    use cozy_chess::{File, Piece, Square};
    let mut a = acc.clone();
    let stm = board.side_to_move();
    let them = match stm {
        Color::White => Color::Black,
        Color::Black => Color::White,
    };
    let moving = moving as usize;

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
        if let Some((v, vsq)) = victim {
            row_sub(&mut a, net, them, v as usize, vsq as usize);
        }
        let placed = mv.promotion.map(|p| p as usize).unwrap_or(moving);
        row_add(&mut a, net, stm, placed, mv.to as usize);
    }
    a
}

/// Gate for the 2026-09-09 AVX2 forward: it must return the SAME INTEGER as the scalar
/// reference, not merely a close one, because the bench signature and every stored TT score
/// depend on the exact value.
///
/// Random nets rather than the shipped one, deliberately: the shipped weights are small and
/// would never approach the vpmaddubsw saturation boundary, so a test against them would pass
/// while proving nothing about the case that could actually break.
#[cfg(all(test, target_arch = "x86_64", target_feature = "avx2"))]
mod avx2_tests {
    use super::*;

    fn xorshift(seed: &mut u64) -> u64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        *seed
    }

    /// A net whose weights and biases span the full quantized ranges.
    fn random_net(seed: &mut u64, input_dim: usize) -> Network {
        let n = |s: &mut u64, lo: i32, hi: i32| lo + (xorshift(s) % (hi - lo + 1) as u64) as i32;
        Network {
            ft: (0..input_dim * ACC).map(|_| n(seed, -512, 512) as i16).collect(),
            ft_bias: (0..ACC).map(|_| n(seed, -127, 127) as i16).collect(),
            w1: (0..HIDDEN * 2 * ACC).map(|_| n(seed, -128, 127) as i8).collect(),
            b1: (0..HIDDEN).map(|_| n(seed, -8128, 8128)).collect(),
            w2: (0..HIDDEN).map(|_| n(seed, -128, 127)).collect(),
            b2: n(seed, -8128, 8128),
            input_dim,
        }
    }

    #[test]
    fn avx2_matches_scalar_on_random_nets() {
        let mut seed = 0x243F_6A88_85A3_08D3u64;
        for _ in 0..8 {
            let net = random_net(&mut seed, 64);
            for _ in 0..256 {
                let mut w = [0i16; ACC];
                let mut b = [0i16; ACC];
                for i in 0..ACC {
                    // spans well past the 0..127 clamp on both sides, so the clamp path is
                    // exercised as much as the multiply path
                    w[i] = (xorshift(&mut seed) % 700) as i16 - 300;
                    b[i] = (xorshift(&mut seed) % 700) as i16 - 300;
                }
                for stm in [Color::White, Color::Black] {
                    assert_eq!(
                        forward_scalar(&net, &w, &b, stm),
                        unsafe { forward_avx2(&net, &w, &b, stm) },
                        "AVX2 forward disagrees with the reference"
                    );
                }
            }
        }
    }

    /// The saturation boundary itself: every activation at its maximum (127) against every
    /// weight at its most negative (-128) is the largest magnitude vpmaddubsw can be asked to
    /// produce -- 2 * 127 * 128 = 32512, which must still fit i16. If a future ACC/FT_Q change
    /// broke that bound, this is the test that would catch it.
    #[test]
    fn worst_case_pair_does_not_saturate() {
        let mut seed = 1;
        let mut net = random_net(&mut seed, 8);
        for w in net.w1.iter_mut() {
            *w = i8::MIN;
        }
        let sat = [i16::MAX; ACC]; // clamps to FT_Q = 127, the maximum activation
        assert_eq!(
            forward_scalar(&net, &sat, &sat, Color::White),
            unsafe { forward_avx2(&net, &sat, &sat, Color::White) }
        );
        net.w1.iter_mut().for_each(|w| *w = i8::MAX);
        assert_eq!(
            forward_scalar(&net, &sat, &sat, Color::White),
            unsafe { forward_avx2(&net, &sat, &sat, Color::White) }
        );
        assert!(2 * (FT_Q as i32) * 128 < i16::MAX as i32 + 1, "saturation bound broken");
    }
}
