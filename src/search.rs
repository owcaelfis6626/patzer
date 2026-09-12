//! Stage-1.5 classical search: iterative deepening, aspiration windows, PVS negamax (fail-soft),
//! transposition table, quiescence with in-check evasions + SEE pruning, null-move pruning,
//! late move pruning (move-count schedule widened when "improving"),
//! reverse futility, LMR, ordering = TT move > SEE-winning captures (MVV-LVA) > queen promos >
//! killers > countermove > (butterfly history + continuation history) > SEE-losing captures.
//! Continuation history (Stage 4): (piece,to) of the 1-ply and 2-ply predecessors index a
//! histogram summed into the residual quiet score; same gravity update as butterfly history.
//! Single thread.
//!
//! MOVE-ORDERING WORK TRIED AND REJECTED, 2026-09-10. Recorded because every one of these ideas
//! is obviously right and someone will try them again. Bench tree against the champion's
//! 284,537 nodes (depth 11) / 1,271,827 (depth 14):
//!
//!   STAGED MOVE GENERATION. Built properly -- TT move yielded with NO generation at all via
//!   `Board::is_legal`, then noisy (captures + queen promotions), then quiets, then the losing
//!   captures held back from the noisy stage; the two destination masks partition every legal
//!   move exactly once and the bucket order is preserved. Passed every gate. Its premise was
//!   measured first and held: cozy's generate_moves costs ~271 cycles and constructing every
//!   Move on top costs ~5 more, so generation is nearly free next to scoring (~2580 of the
//!   ~2860 cycles that region takes). The predicted speed arrived -- 1565 knps against 1397,
//!   +12% -- and a 13.5% bigger tree ate all of it (910 ms vs 923 ms to depth 14). Divergence
//!   begins at DEPTH 2 (+0.67%) and compounds to +22% by depth 11, the signature of a small
//!   per-node ordering change rather than a bug. Stable sorts within each stage recovered
//!   almost none of it.
//!
//!   Then three attempts to fix what staging exposed -- that most quiets have history EXACTLY
//!   ZERO and so form one large tie group whose order is arbitrary:
//!
//!       PeSTO piece-square prior on quiets      +0.67% / +9.50%
//!       quiet checks ordered at 70_000          +25.3% / +39.5%
//!       quiet checks as an additive +1024       +19.8% / +28.6%
//!                                  +4096        +33.8% / +22.9%
//!                                 +16384        +22.5% / +55.9%
//!
//!   ALL WORSE. So "the zero-history tie group is costing Elo" is NOT a supported claim: it was
//!   inferred from the staged result and has now resisted three independent probes at five
//!   settings. Either the existing order is already near what these heuristics would impose, or
//!   history carries more signal near zero than it appears to. Do not treat it as known
//!   headroom.
//!
//! WHAT THE EXERCISE DID YIELD: the generator already hands over `pm.piece`, so the per-move
//! `board.piece_on(mv.from)` rescan -- up to six bitboard tests, ~33 times a node, at four call
//! sites -- was pure waste. Bit-exact (signature 284537 unchanged), +3.6% nps, and the ranges
//! over 12 runs do not overlap: champion 1346-1376 knps against 1394-1427.
//!
//! The standing diagnosis these all came from: measured against stash-v26 on identical
//! positions, patzer reaches depth 21.92 where stash reaches 29.38 at the same 1.35 s/move,
//! decomposing into EBF 1.689 vs 1.510 and nps 1.26M vs 2.60M. patzer still wins that match by
//! >100 Elo, so the EVAL carries this engine and the SEARCH is the weaker half.

use crate::eval::{evaluate, piece_val};
use crate::nnue;
use crate::tt::{pack, BOUND_EXACT, BOUND_LOWER, BOUND_UPPER, TT};
use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves,
    BitBoard, Board, Color, Move, Piece, Rank, Square,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const MATE: i32 = 32_000;
pub const INF: i32 = 32_500;
const MATE_BOUND: i32 = MATE - 512;
const MAX_PLY: usize = 128;
const CONT_PT: usize = 6 * 64; // continuation-history (piece,to) index space = 384
// Each block gets one extra ZERO row (index CONT_PT) as the "no predecessor / null move"
// sentinel, so cont_score can read unconditionally instead of branching on Option + NULL_CONT
// at every quiet move. The extra row is never written (cont_bonus skips it), so it stays zero.
const CONT_STRIDE: usize = (CONT_PT + 1) * CONT_PT;
const LMP_MAX_DEPTH: i32 = 8; // above this the move-count schedule is too blunt to be safe
const IIR_MIN_DEPTH: i32 = 4; // 0 disables internal iterative reduction
// REVERTED 2026-08-14, and the reason CORRECTED 2026-08-19. The original note here said
// "+25 in self-play, ~-35 vs Stash". The -35 was not real: it compared a correctly-measured
// SE arm against a -6.95 pre-SE baseline that was a 400-game outlier (that binary has since
// measured -23.78, -23.30 and -24.24). A 6000-game PAIRED run against Stash v26 puts the
// true difference at -4.89 Elo, 95% CI [-18.1, +8.3] -- indistinguishable from zero.
// What actually justifies SE=0: the +25.08 +- 13.78 self-play gain DOES NOT TRANSFER
// (upper bound +8.3), so a 2.5x bigger tree buys nothing measurable outside self-play.
// Powered to ~+-13 Elo; a smaller real effect is not excluded. See campaign.jsonl
// "se_paired_external".
const SE_MIN_DEPTH: i32 = 0;
// Singular-extension FAMILY, 2026-09-12. The old code returned at most +1. Modern engines
// (SF 18, Reckless, PlentyChess, Alexandria) instead: scale the singular margin with depth,
// add a 2nd/3rd ply when the excluded search fails even lower (double/triple extension),
// prune the whole subtree when several moves fail high (multi-cut), and REDUCE the TT move
// when the evidence says it is not singular (negative extension). All of it is gated by
// SE_MIN_DEPTH, which is 0, so the champion tree is untouched until an SPRT turns it on.
// MARGINS ARE SF-SHAPED AND ARE THE THING TO RE-TUNE once enabled.
const SE_MARGIN: i32 = 60;         // singularBeta = tt_score - SE_MARGIN*depth/60
const SE_DOUBLE: bool = true;      // 2nd ply when value < singularBeta - 4*depth/60
const SE_TRIPLE: bool = true;      // 3rd ply when value < singularBeta - 73*depth/60
const SE_DOUBLE_MARGIN: i32 = 4;
const SE_TRIPLE_MARGIN: i32 = 73;
const SE_MULTICUT: bool = true;    // several moves fail high -> the node is not singular
const SE_NEGATIVE: bool = true;    // reduce the TT move when it is not singular
const LMR_TWEAKS: bool = true; // PV/improving adjustments to the LMR reduction

// ---- 2026-09-09 search batch. Each gate is independent so a failed SPRT can be bisected
// without a rebuild matrix. All default OFF; the campaign turns them on one arm at a time.
const QS_TT: bool = true;       // transposition probe + store in quiescence
const FUTILITY: bool = true;   // forward futility pruning of quiets near the horizon
const SEE_PRUNE: bool = true;  // SEE pruning of losing captures and quiets in negamax
const MATE_DIST: bool = true;  // mate-distance pruning at the top of negamax
const CAPT_HIST: bool = true;  // history table for capture ordering, beyond MVV-LVA
// Scale the LMR reduction by the move's history score. THE DIVISOR IS NOT A GUESS: the first
// version used 8192 and was measured completely inert -- bench signature identical to the
// feature being off, because at LMR sites |history| never reaches 8192. Distribution measured
// over the bench suite (n = 78k / 539k / 1.37M sites):
//
//     |h| >= 4096 :  0.0% at depth 11,  0.5% at depth 14,  1.4% at depth 16
//     |h| >=  256 :  2.4% at depth 11, 11.6% at depth 14, 19.4% at depth 16
//
// so the tables fill with depth and the signature depth of 11 badly understates what a 40/15
// search will hold. 512 makes the term bite on roughly a tenth of LMR'd moves at depth 16 and
// more above that; it is exposed to SPSA rather than trusted.
const HIST_LMR: bool = true;
// Reuse the ordering score as the HIST_LMR input instead of reloading the history tables.
//
// NOT BIT-EXACT, AND THAT IS THE FINDING. A history-scored quiet's ordering score IS
// `history + cont_score` at the moment it is computed -- but those tables are MUTATED during
// this very node's move loop, by the child searches it launches (cont_bonus, and the history
// updates on a fail-high). The original code therefore reads a FRESH value at the LMR site,
// while reusing the score reads an entry-time SNAPSHOT. Measured: signature 270698 against the
// champion's 284537. Defensible as a design choice -- ordering-time consistency is arguably what
// you want -- but it changes the tree and needs an SPRT, so it is not the free win it appears to
// be. Left gated rather than discarded; tag::IS_HIST_SCORE carries the distinction for free.
const HIST_LMR_REUSE: bool = false;
const RAZOR: bool = false;      // drop to qsearch when the static eval is far below alpha
const NMP_VERIFY: bool = false; // verify a null-move cutoff with a null-disabled re-search
const NMP_VERIFY_DEPTH: i32 = 8;// only above this depth; below it the re-search costs more than it saves
const CAPT_LMR: bool = true;   // reduce late CAPTURES too, not only late quiets
// Stop extending checks once a line has run far past the root depth.
//
// MEASURED 2026-09-09 AND CURRENTLY NOT WORTH TESTING. Together with NMP_VERIFY this moves the
// tree by +0.22% / -1.86% / +1.86% at bench depths 11 / 14 / 17 against an otherwise identical
// binary -- a near-no-op at every depth reachable on this hardware, and non-monotone, so it is
// not "a feature that engages with depth" either. The reason is arithmetic: the cap is
// `ply < 2 * root_depth`, so at depth 17 a check line would have to run 34 plies before it ever
// binds, which essentially never happens. Making it bite needs a real budget -- a per-path count
// of check extensions with a hard limit -- not a ply threshold scaled off the root depth.
// A +/-2% tree change cannot produce Elo an SPRT of affordable size would see, so this is parked
// rather than measured. (The measurement needed a champion-equivalent binary REBUILT with the
// PATZER_BENCH_DEPTH override: bin/patzer-champ predates it and silently ran depth 11 whatever
// the variable said, which made the first version of this comparison meaningless.)
const CHECK_EXT_CAP: bool = false;
// Bigger history bonuses. MEASURED MOTIVATION, not taste: the |history| distribution at LMR
// sites is 89.9% below 64 at bench depth 11 and still 45.6% below 64 at depth 16 (see the
// HIST_LMR note). The tables are barely populated, so they can hardly discriminate between the
// late quiets they are asked to rank. The gravity update saturates at 16384 either way -- the
// bonus only sets how FAST an entry gets there, and at (depth*depth).min(400) it is slow.
const HIST_BONUS: bool = true;
// Scale the reverse-futility margin by `improving`. A node whose eval is going the wrong way
// deserves to be written off sooner than one that is climbing; RFP currently treats them alike.
const RFP_IMPROVING: bool = true;
// Let a null move reduce more when the static eval is far above beta -- the further above,
// the more certain the pass is safe.
const NMP_EVAL_R: bool = true;
// ProbCut. If a capture already beats a RAISED beta at reduced depth, the node almost certainly
// beats the real beta at full depth. Verified with a qsearch first so the reduced-depth search is
// only paid for on candidates that already look like they cut.
const PROBCUT: bool = true;
const PROBCUT_MARGIN: i32 = 200; // how far above beta the raised bound sits
const PROBCUT_MIN_DEPTH: i32 = 5;
const PROBCUT_REDUCTION: i32 = 4;
// Node-type awareness. A "cut node" is one the search EXPECTS to fail high: the child of an
// all-node, or any null-window child of a node that already failed to raise alpha. The
// expectation is usually right, so a cut node deserves to be searched more cheaply -- if it is
// going to fail high anyway, spending full depth on the moves that will not cause it is waste.
// Stockfish uses this in roughly a dozen places; the two that carry most of it are LMR (reduce
// harder) and IIR (reduce harder when there is no table move to order by).
const CUT_NODE: bool = true;
const CUT_LMR_BONUS: i32 = 2;   // extra LMR plies at an expected-fail-high node
const CUT_IIR_MIN_DEPTH: i32 = 7; // extra IIR reduction at a cut node with no TT move
// Prune late quiets the history actively dislikes, before the board clone. LMP already caps how
// MANY quiets are searched; this cuts the ones the tables have specific evidence against, which
// is a different screen and fires at shallower depth than LMP's move-count schedule reaches.
// MEASURED 2026-09-09 AND PARKED, together with KILLER_CLEAR and CONT_EXTRA, for ONE SHARED
// REASON: this engine's history values are an order of magnitude smaller than in the engines
// these heuristics are borrowed from, so every threshold written against the usual scale is
// either inert or wrong. Threshold sweep, bench tree vs 436352:
//
//     -2000*d  -500*d  -200*d   INERT -- never fires, signature bit-identical
//     -100*d   +0.02%
//      -50*d   +6.87%   -25*d  +8.37%   <- fires, and makes the tree BIGGER
//
// A pruning rule that costs nodes is pruning moves that were about to cause cutoffs. The root
// cause is the table scale: |history| at LMR sites is 89.9% below 64 and never exceeds 4096
// (measured), because the bonus is (depth*depth).min(400) against a gravity divisor of 16384.
// HIST_BONUS (in the K_all candidate) raises that bonus 4x. IF K_all IS PROMOTED, RE-MEASURE
// THE DISTRIBUTION AND RE-CALIBRATE THIS THRESHOLD AND HIST_LMR_DIV BEFORE RETRYING -- the
// numbers here describe the old, smaller tables and will not transfer.
const HIST_PRUNE: bool = false;
const HIST_PRUNE_MAX_DEPTH: i32 = 4;
const HIST_PRUNE_THRESHOLD: i32 = -2000;
// Clear the killers two plies ahead. MEASURED 2026-09-09: +41.6% bench tree, and parked.
// The reasoning that motivated it is wrong for this search. A node at ply P clears killers[P+2],
// then searches children at P+1 whose own children at P+2 find the slot empty -- so within any
// subtree the killers at P+2 are always blank on first use, and the heuristic is destroyed
// rather than refreshed. Stockfish gets away with the same line because of where it sits
// relative to its stack handling; here it is a straight loss.
const KILLER_CLEAR: bool = false;
// Continuation history depth. The 1- and 2-ply predecessors answer "does this move work after
// that one"; the 4- and 6-ply ones answer "does it work in this kind of plan", which is a
// different and largely independent question. Stockfish keeps 1, 2, 3, 4 and 6.
// With CONT_EXTRA off only the first two blocks are ever touched, so the extra allocation is
// never read and the behaviour is identical.
// MEASURED 2026-09-09, EXCLUDED FROM THE COMBINED CANDIDATE. Turning this on inflates the
// bench tree 57% (436352 -> 685759) at equal depth. That is a large bill for a reordering, and
// the naive implementation here is the likely reason: the four blocks are summed with EQUAL
// weight into one quiet score, so the extra two mostly-empty tables add noise early, and a
// maximal 4-block sum (~82k) can outrank a killer (80k) outright -- an ordering inversion the
// two-block version cannot produce. Stockfish weights the blocks separately and consults them
// in pruning decisions rather than folding them into one number. Worth revisiting as weighted
// blocks; not worth games as an equal-weight sum.
// FOLLOW-UP 2026-09-09: retried with per-block divisors [1,1,4,4] so the distant blocks only
// refine. Still +42.5% tree (from +57%), so damping the READ is not enough -- the bonus written
// to all four blocks is identical, and the distant tables were too sparse to rank anything.
//
// REVERSED 2026-09-09, LATER THE SAME DAY, AND THE REVERSAL IS THE POINT. Once HIST_BONUS was
// promoted (4x the bonus, so |h| >= 256 went from 9.3% to 30.1% of LMR sites) the SAME code
// measures -10.3% tree instead of +42.5%. Nothing about this feature changed; the tables it
// reads did. THE EARLIER KILL WAS AN ARTEFACT OF TEST ORDER, not a property of the feature --
// four sparse blocks cannot rank moves, four populated ones can. Any history-derived heuristic
// measured against the pre-HIST_BONUS tables has to be re-measured before it is believed.
const CONT_EXTRA: bool = false;
const CONT_PLIES: [usize; 4] = [1, 2, 4, 6]; // how many plies back each block indexes
// Per-block divisors. The 1- and 2-ply blocks answer "does this move work after that one" and
// carry full weight; the 4- and 6-ply blocks answer the vaguer "does it suit this kind of plan"
// and are damped so they refine the ranking instead of dominating it. The equal-weight version
// cost +57% tree (see CONT_EXTRA).
const CONT_WEIGHT_DIV: [i32; 4] = [1, 1, 4, 4];
const CONT_BLOCKS: usize = 4;
const RAZOR_MARGIN: i32 = 300;  // per ply of remaining depth
const RAZOR_MAX_DEPTH: i32 = 3;
const CAPT_SIZE: usize = 2 * 384 * 6; // [stm][piece*64+to][victim]
const FUT_MARGIN: i32 = 100;    // futility margin per ply of remaining depth
const FUT_MAX_DEPTH: i32 = 6;   // above this the static eval is too stale to prune on
const SEE_Q_MARGIN: i32 = -50;  // quiets worse than this by SEE are cut, scaled by depth
const SEE_C_MARGIN: i32 = -100; // losing captures worse than this by SEE are cut, x depth
const NULL_CONT: usize = usize::MAX; // sentinel: no continuation across a null move

// Correction history. The static eval is systematically wrong in whole CLASSES of position --
// a structure the net habitually over- or under-rates -- and the search discovers this every
// time it returns a score far from the eval it started with. That discrepancy is thrown away
// today. Here it is recorded against the pawn structure (which is what makes two positions
// "the same kind of position") and added back to the static eval next time.
//
// Indexed by pawn structure ONLY, deliberately: it has to generalise across positions that
// share a character, so a full board key would never hit twice. Costs one hash + one array
// read per node -- unlike the accumulator this sits BESIDE the net, so it does not disturb
// incremental updates.
const CORR_HIST: bool = true; // correction history on the static eval
const CORR_SIZE: usize = 16_384; // 2^14 pawn-structure buckets per side
const CORR_GRAIN: i32 = 256; // entries kept in 1/256 pawn units so the EMA stays smooth in i32
const CORR_MAX: i32 = 128 * CORR_GRAIN; // never shift the eval by more than ~128cp
// Apply the correction at the qsearch stand-pat as well as in negamax.
//
// WHY THIS IS A SEPARATE TOGGLE. CORR_HIST measured +6.41 +- 13.29 over 1300 games -- a null,
// but with an identified hole: the correction was applied in negamax ONLY, and most nodes in
// this search are qsearch nodes, so most static evals never saw it. That makes the null a
// verdict on the deployment, not on the technique.
//
// APPLY, BUT DO NOT UPDATE. qsearch has no meaningful depth -- the EMA weight is depth-scaled
// (`w = depth.clamp(1, CORR_W_MAX)`) and every qsearch sample would enter at weight 1 while
// being the least reliable evidence available. So the table is still learned from negamax
// nodes only, and merely consulted more often.
const CORR_QSEARCH: bool = true;
const CORR_W: i32 = 16; // EMA horizon; a single sample can take at most half the entry
const CORR_W_MAX: i32 = CORR_W / 2;

// Volatility-aware time management. Measured on 1,018,380 samples from the self-play PGN
// corpus (games/jump_test.py): windowed realised eval volatility predicts the size of the
// NEXT eval jump at AUC 0.81, and adding depth / time / |eval| on top lifts that only to
// 0.82 -- past volatility is essentially the whole signal, so no learned head is needed.
// VOL_W recent own-side root scores are averaged; the soft time limit is scaled between
// VOL_MIN and VOL_MAX as that average runs from VOL_LO to VOL_HI pawns.
//
// The bench signature CANNOT gate this: bench runs at fixed depth, so time management never
// engages and the signature is identical with the toggle on and off. `search::vol_tm_tests`
// is the gate instead.
const VOL_TM: bool = false;
const VOL_W: usize = 8;
const VOL_LO: f64 = 0.10; // pawns: calm
const VOL_HI: f64 = 0.60; // pawns: the corpus's "big jump" tertile boundary was 0.65
const VOL_MIN: f64 = 0.80; // spend 20% less when the eval has been flat
const VOL_MAX: f64 = 1.60; // spend 60% more when it has been swinging

// ---------------------------------------------------------------- tunable parameters
// These six constants govern how aggressively the search prunes, and none of them has ever
// been measured -- they are the values that happened to work when each feature was written.
// Tuning them needs the engine to vary them at RUNTIME (SPSA changes parameters every few
// games; a rebuild per iteration would make it hopeless), but a runtime read in the hot path
// would cost nps in the shipped binary.
//
// So: behind `--features tune` they become atomics settable over UCI. In a normal build they
// stay `const`, the accessors inline away to literals, and the bench signature is unchanged
// (795654). Tuned values get written back as constants afterwards -- the tune build is a
// measuring instrument, not something that ships.
#[cfg(not(feature = "tune"))]
pub mod tune {
    pub const RFP_MARGIN: i32 = 120;   // reverse-futility margin per ply
    pub const NMP_BASE: i32 = 3;       // null-move reduction, constant part
    pub const NMP_DIV: i32 = 5;        // null-move reduction, depth divisor
    // SWEPT, THEN REFUTED BY SPRT -- both halves matter, 2026-09-10.
    //
    // Measured against 250 real games vs stash-v26: patzer reaches mean depth 21.92 where stash
    // reaches 29.38 at the same 1.35 s/move, decomposing into EBF 1.689 vs 1.510 and nps 1.26M
    // vs 2.60M. Sweeping these constants for depth-at-fixed-time found exactly one that moves
    // the tree. Bench tree against the champion's 284537:
    //
    //     LMR_DIV 225->180   238763  (-16.1%)    LMR_DIV 225->150   222572  (-21.8%)
    //     LMR_BASE 75->100   280086   (-1.6%)    both               235557  (-17.2%)
    //     NMP_BASE 3->4      297199   (+4.5%)    LMP_BASE 3->2      317956  (+11.8%)
    //
    // NMP_BASE and LMP_BASE are nominally MORE aggressive and make the tree BIGGER -- they prune
    // moves that were about to cause cutoffs, so the node is searched more widely instead. All
    // four together FAIL the `mates` gate while each alone passes: over-pruning caught by a
    // correctness gate rather than a wasted SPRT.
    //
    // AND THEN THE SPRT KILLED THE PRESCRIPTION. LMR_DIV=180 -- the "one live lever" -- measured
    // -26.86 +/- 20.63 over 350 games at 10+0.1, CI entirely below zero, bundled with a bit-exact
    // speed win that could only have helped. Proof the speed win was not the cause:
    // patzer-O_lmrdiv180 and patzer-R_pmpiece_lmr180 bench the IDENTICAL 238763 tree, and R ran
    // 1423 knps against O's 1334 -- strictly faster on the same tree, and it still lost. So
    // LMR_DIV=180 costs roughly -35 Elo by itself.
    //
    // The EBF decomposition is sound as a DESCRIPTION and wrong as a PRESCRIPTION: pruning harder
    // does close the gap toward stash's 1.510, and loses Elo doing it. Stash's lower EBF comes
    // from ordering or eval shape, not from timider reductions here. DO NOT tune these toward a
    // smaller tree -- and note SPSA optimises SCORE, not tree size, so it is unaffected by this.
    //
    // THIRD TIME IN ONE DAY THAT BENCH TREE SIZE MISLED A DECISION:
    //     CONT_EXTRA   -10.3% tree ->  -0.87 +/- 19.4  (null)
    //     HIST_PRUNE    -5.4% tree ->  +1.74 +/- 21.3  (null)
    //     LMR_DIV=180  -16.1% tree -> -26.86 +/- 20.6  (negative)
    // Tree size is ANTI-CORRELATED with Elo when the shrinkage comes from pruning harder. Use the
    // bench signature for "did this change anything", never for "is this better".
    pub const LMP_BASE: i32 = 3;       // late-move-pruning schedule offset
    pub const LMR_BASE: i32 = 75;      // LMR table intercept, x100
    pub const LMR_DIV: i32 = 225;      // LMR table divisor, x100
    pub const HIST_LMR_DIV: i32 = 512; // history units per ply of LMR adjustment
    pub const HIST_BONUS_MUL: i32 = 4;   // history bonus = mul*depth^2, capped
    pub const HIST_BONUS_MAX: i32 = 1600;
    #[inline(always)] pub fn rfp_margin() -> i32 { RFP_MARGIN }
    #[inline(always)] pub fn nmp_base() -> i32 { NMP_BASE }
    #[inline(always)] pub fn nmp_div() -> i32 { NMP_DIV }
    #[inline(always)] pub fn lmp_base() -> i32 { LMP_BASE }
    #[inline(always)] pub fn lmr_base() -> i32 { LMR_BASE }
    #[inline(always)] pub fn lmr_div() -> i32 { LMR_DIV }
    #[inline(always)] pub fn hist_lmr_div() -> i32 { HIST_LMR_DIV }
    #[inline(always)] pub fn hist_bonus_mul() -> i32 { HIST_BONUS_MUL }
    #[inline(always)] pub fn hist_bonus_max() -> i32 { HIST_BONUS_MAX }
}

#[cfg(feature = "tune")]
pub mod tune {
    use std::sync::atomic::{AtomicI32, Ordering};
    pub static RFP_MARGIN: AtomicI32 = AtomicI32::new(120);
    pub static NMP_BASE: AtomicI32 = AtomicI32::new(3);
    pub static NMP_DIV: AtomicI32 = AtomicI32::new(5);
    pub static LMP_BASE: AtomicI32 = AtomicI32::new(3);
    pub static LMR_BASE: AtomicI32 = AtomicI32::new(75);
    pub static LMR_DIV: AtomicI32 = AtomicI32::new(225);
    pub static HIST_LMR_DIV: AtomicI32 = AtomicI32::new(512);
    pub static HIST_BONUS_MUL: AtomicI32 = AtomicI32::new(4);
    pub static HIST_BONUS_MAX: AtomicI32 = AtomicI32::new(1600);
    #[inline] pub fn rfp_margin() -> i32 { RFP_MARGIN.load(Ordering::Relaxed) }
    #[inline] pub fn nmp_base() -> i32 { NMP_BASE.load(Ordering::Relaxed) }
    #[inline] pub fn nmp_div() -> i32 { NMP_DIV.load(Ordering::Relaxed).max(1) }
    #[inline] pub fn lmp_base() -> i32 { LMP_BASE.load(Ordering::Relaxed) }
    #[inline] pub fn lmr_base() -> i32 { LMR_BASE.load(Ordering::Relaxed) }
    #[inline] pub fn lmr_div() -> i32 { LMR_DIV.load(Ordering::Relaxed).max(1) }
    #[inline] pub fn hist_lmr_div() -> i32 { HIST_LMR_DIV.load(Ordering::Relaxed).max(1) }
    #[inline] pub fn hist_bonus_mul() -> i32 { HIST_BONUS_MUL.load(Ordering::Relaxed).max(1) }
    #[inline] pub fn hist_bonus_max() -> i32 { HIST_BONUS_MAX.load(Ordering::Relaxed).max(1) }

    /// (uci name, default, min, max) -- the driver reads this to build its parameter set.
    pub const PARAMS: &[(&str, i32, i32, i32)] = &[
        ("RfpMargin", 120,  40, 240),
        ("NmpBase",     3,   1,   6),
        ("NmpDiv",      5,   2,  12),
        ("LmpBase",     3,   1,   8),
        ("LmrBase",    75,  20, 150),
        ("LmrDiv",    225, 120, 400),
        ("HistLmrDiv", 512, 64, 4096),
        ("HistBonusMul",  4,  1,  32),
        ("HistBonusMax", 1600, 200, 8000),
    ];

    pub fn set(name: &str, v: i32) -> bool {
        let t = |a: &AtomicI32| { a.store(v, Ordering::Relaxed); true };
        match name.to_ascii_lowercase().as_str() {
            "rfpmargin" => t(&RFP_MARGIN),
            "nmpbase"   => t(&NMP_BASE),
            "nmpdiv"    => t(&NMP_DIV),
            "lmpbase"   => t(&LMP_BASE),
            "lmrbase"   => t(&LMR_BASE),
            "lmrdiv"    => t(&LMR_DIV),
            "histlmrdiv" => t(&HIST_LMR_DIV),
            "histbonusmul" => t(&HIST_BONUS_MUL),
            "histbonusmax" => t(&HIST_BONUS_MAX),
            _ => false,
        }
    }
}

/// History-scale instrumentation, behind `--features instrument` so the shipped binary pays
/// nothing for it.
///
/// WHY THIS IS PERMANENT RATHER THAN A SCRATCH PATCH. Three separate heuristics were written
/// against the history scale of OTHER engines and failed here for that one reason -- HIST_LMR
/// was completely inert with a divisor of 8192, HIST_PRUNE never fired above a threshold of
/// -100*depth, and both were only diagnosed by measuring this distribution. Any change to the
/// history bonus (HIST_BONUS, or SPSA moving HistBonusMul/Max) moves the whole scale, and every
/// threshold denominated in history units has to be re-read against it afterwards.
///
///   cargo build --release --features instrument && patzer histdump [depth]
#[cfg(feature = "instrument")]
pub mod instrument {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const EDGES: [i32; 9] = [64, 256, 1024, 2048, 4096, 8192, 16384, 32768, i32::MAX];
    pub static LMR_HIST: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

    #[inline]
    pub fn record(h: i32) {
        let a = h.unsigned_abs() as i32;
        let mut i = 0;
        while i < 9 && a >= EDGES[i] {
            i += 1;
        }
        LMR_HIST[i].fetch_add(1, Ordering::Relaxed);
    }

    // ---- move-ordering quality -------------------------------------------------------
    // The fraction of fail-high nodes that cut on the FIRST move is what actually drives the
    // effective branching factor, and it is the one number this project has never measured.
    // We know EBF 1.689 against stash-v26's 1.510 and depth 21.9 vs 29.4, and we know that
    // pruning harder to close the gap LOSES Elo (LMR_DIV 180: -26.86). So the gap is ordering
    // or eval shape, and this separates them: a well-ordered search cuts on move 1 ~90-95% of
    // the time. The bucket histogram then says WHICH ordering source is underperforming.
    pub const BUCKETS: [&str; 6] = ["TT move", "winning capture", "queen promo",
                                    "killer/counter", "quiet (history)", "losing capture"];
    pub static FH_INDEX: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
    pub static FH_BUCKET: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

    /// `ord_score` is the ORDERING score from the move buffer (not the search score), so the
    /// bands below are the scoring closure's own: 1_000_000 TT, 100_000+mvvlva+ch winning
    /// capture (min 99_539), 95_000 queen promo, 80_000/78_000 killer/countermove, then
    /// history-scored quiets and losing captures, which overlap in sign and are separated by
    /// the tag's CAPTURE bit rather than by value.
    #[inline]
    pub fn record_cutoff(idx: usize, ord_score: i32, tag_bits: u8) {
        FH_INDEX[idx.min(4)].fetch_add(1, Ordering::Relaxed);
        let b = if ord_score >= 1_000_000 {
            0
        } else if ord_score >= 99_000 {
            1
        } else if ord_score >= 94_000 {
            2
        } else if ord_score >= 77_000 {
            3
        } else if super::tag::is_capture(tag_bits) {
            5
        } else {
            4
        };
        FH_BUCKET[b].fetch_add(1, Ordering::Relaxed);
    }

    pub fn report_ordering() {
        let tot: u64 = FH_INDEX.iter().map(|b| b.load(Ordering::Relaxed)).sum();
        if tot == 0 {
            println!("  no fail-high nodes recorded");
            return;
        }
        let first = FH_INDEX[0].load(Ordering::Relaxed);
        println!("  MOVE-ORDERING QUALITY (n={tot} fail-high nodes in negamax)");
        println!("    first-move cutoff rate: {:.2}%   <- the EBF driver",
                 100.0 * first as f64 / tot as f64);
        let labels = ["1st move", "2nd", "3rd", "4th", "5th+"];
        let mut cum = 0u64;
        for (i, l) in labels.iter().enumerate() {
            let c = FH_INDEX[i].load(Ordering::Relaxed);
            cum += c;
            println!("    cut on {:<9} {:>11}  {:5.1}%   cum {:5.1}%",
                     l, c, 100.0 * c as f64 / tot as f64, 100.0 * cum as f64 / tot as f64);
        }
        println!("  WHICH ORDERING SOURCE PRODUCED THE CUT");
        for (i, name) in BUCKETS.iter().enumerate() {
            let c = FH_BUCKET[i].load(Ordering::Relaxed);
            println!("    {:<16} {:>11}  {:5.1}%",
                     name, c, 100.0 * c as f64 / tot as f64);
        }
    }

    pub fn report() {
        report_ordering();
        let tot: u64 = LMR_HIST.iter().map(|b| b.load(Ordering::Relaxed)).sum();
        if tot == 0 {
            println!("  no LMR sites recorded");
            return;
        }
        println!("  |history + continuation| at LMR sites (n={tot}):");
        let mut lo = 0i32;
        let mut cum = 0u64;
        for (i, e) in EDGES.iter().enumerate() {
            let c = LMR_HIST[i].load(Ordering::Relaxed);
            cum += c;
            let hi = if *e == i32::MAX { "inf".to_string() } else { e.to_string() };
            println!("    {:>6}..{:<8} {:>11}  {:5.1}%   cum {:5.1}%",
                     lo, hi, c, 100.0 * c as f64 / tot as f64, 100.0 * cum as f64 / tot as f64);
            lo = *e;
        }
    }
}

/// Dead draws by material, conservatively: K vs K, K+minor vs K, and K+N+N vs K (which cannot
/// be forced). Deliberately NOT K+B vs K+B -- that depends on square colour and can be won when
/// the bishops are same-coloured is false, but the general case needs care and a wrong "draw"
/// throws away real wins. Anything with a pawn, rook or queen returns immediately.
#[inline]
fn insufficient_material(board: &Board) -> bool {
    let occ = board.occupied();
    if occ.len() > 4 {
        return false;
    }
    if !board.pieces(Piece::Pawn).is_empty()
        || !board.pieces(Piece::Rook).is_empty()
        || !board.pieces(Piece::Queen).is_empty()
    {
        return false;
    }
    let knights = board.pieces(Piece::Knight).len();
    let bishops = board.pieces(Piece::Bishop).len();
    match occ.len() {
        2 => true,                                  // K vs K
        3 => knights + bishops == 1,                // K + one minor vs K
        4 => knights == 2 && bishops == 0           // K+N+N vs K: not forceable
            && (board.pieces(Piece::Knight) & board.colors(Color::White)).len() != 1,
        _ => false,
    }
}

fn opp(c: Color) -> Color {
    match c {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

/// The piece `mv` captures, or None for a non-capture.
///
/// 2026-08-18 FIX. Every capture test in this file used to be
/// `board.color_on(mv.to) == Some(opp(stm))`, which is FALSE for en passant: the destination
/// square is empty. nnue::acc_update_basic already relied on exactly that fact to detect ep
/// (empty `to` + a diagonal pawn move), so the codebase proved the bug while the search
/// classified ep as a QUIET move -- eligible for late-move pruning and LMR, updating killers
/// and history as a quiet, and, worst of all, never generated in qsearch at all unless in
/// check, leaving quiescence structurally blind to en passant.
///
/// Castling is king-takes-own-rook in cozy-chess, so `color_on(mv.to) == Some(stm)`; it is not
/// a capture and is correctly excluded by the `opp(stm)` test.
#[inline]
// Not called from the SEARCH any more -- every hot site uses `capture_victim_with` with the
// enemy occupancy and en-passant square it already holds. Still the wrapper that ep_tests and
// qmask_tests are written against, and (2026-09-11) the one datagen's record filter uses: a
// hand-rolled `color_on(mv.to)` predicate is FALSE for en passant, which silently admitted
// ep-capture positions into the "quiet" training set.
pub(crate) fn capture_victim(board: &Board, mv: Move) -> Option<Piece> {
    let stm = board.side_to_move();
    capture_victim_with(board, mv, board.colors(opp(stm)), ep_square(board, stm))
}

/// The en-passant destination square, if any.
#[inline]
fn ep_square(board: &Board, stm: Color) -> Option<Square> {
    board
        .en_passant()
        .map(|f| Square::new(f, Rank::Sixth.relative_to(stm)))
}

/// F4 (2026-09-10): the hot-path form, taking the enemy occupancy and en-passant square the
/// caller already has.
///
/// The original cost every QUIET move a `board.color_on(mv.to)` (two bitboard tests) plus a
/// `board.piece_on(mv.from)` to discover it was not a pawn -- and cozy's `piece_on` is a linear
/// scan of up to six bitboards. So a non-pawn quiet paid ~8 tests to be told "not a capture",
/// once per move per node, plus again in qsearch's delta-pruning pass.
///
/// Here a capture costs ONE test, and the en-passant branch is skipped entirely when there is no
/// en-passant square -- which is almost always. `mv.to == ep` already implies the diagonal (the
/// square behind a double-pushed pawn is never reachable by a straight push, since the pusher's
/// own origin would have to be the occupied square the pawn just left), so the file comparison
/// and the `piece_on(mv.to).is_none()` check are both redundant; pawn-ness is one bitboard test.
#[inline]
fn capture_victim_with(
    board: &Board,
    mv: Move,
    enemy: BitBoard,
    ep: Option<Square>,
) -> Option<Piece> {
    if enemy.has(mv.to) {
        return board.piece_on(mv.to);
    }
    if let Some(epsq) = ep {
        if mv.to == epsq && board.pieces(Piece::Pawn).has(mv.from) {
            return Some(Piece::Pawn);
        }
    }
    None
}

// ---------------- static exchange evaluation ----------------
// Swap algorithm with x-ray updates (sliders recomputed on the reduced occupancy each iteration).
// Approximations (standard for ordering): promotions valued as the moving pawn; a king may not
// "capture into" remaining enemy attackers (loop stops there).

#[inline(always)]
fn attackers_to(board: &Board, sq: Square, occ: BitBoard) -> BitBoard {
    let rq = board.pieces(Piece::Rook) | board.pieces(Piece::Queen);
    let bq = board.pieces(Piece::Bishop) | board.pieces(Piece::Queen);
    let pawns = board.pieces(Piece::Pawn);
    ((get_knight_moves(sq) & board.pieces(Piece::Knight))
        | (get_king_moves(sq) & board.pieces(Piece::King))
        | (get_rook_moves(sq, occ) & rq)
        | (get_bishop_moves(sq, occ) & bq)
        | (get_pawn_attacks(sq, Color::White) & pawns & board.colors(Color::Black))
        | (get_pawn_attacks(sq, Color::Black) & pawns & board.colors(Color::White)))
        & occ
}

fn least_valuable(board: &Board, set: BitBoard) -> Option<(Square, Piece)> {
    for p in [
        Piece::Pawn,
        Piece::Knight,
        Piece::Bishop,
        Piece::Rook,
        Piece::Queen,
        Piece::King,
    ] {
        let s = set & board.pieces(p);
        if let Some(sq) = s.into_iter().next() {
            return Some((sq, p));
        }
    }
    None
}

pub fn see(board: &Board, mv: Move) -> i32 {
    match board.piece_on(mv.from) {
        Some(p) => see_with(board, mv, p),
        None => return 0,
    }
}

/// As `see`, but told which piece is moving.
///
/// 2026-09-11: `see` re-derived the attacker with `board.piece_on(mv.from)` on every entry -- a
/// scan of up to six bitboards -- and it is NOT inlined (it has its own symbol and dozens of call
/// sites), at roughly 5-8 calls per node. Every hot caller already holds the piece: the scoring
/// closure has `pm.piece`, qsearch generation has `moved`, SEE_PRUNE has it in the move tag, and
/// ProbCut's generator has `pm.piece`. `see()` stays as the wrapper for the `see` gate and tests,
/// where the `None` guard is still meaningful.
pub fn see_with(board: &Board, mv: Move, attacker: Piece) -> i32 {
    let target = mv.to;
    // en passant: to-square is empty but a pawn is captured
    let mut captured = board.piece_on(target).map(piece_val).unwrap_or(0);
    if captured == 0 && attacker == Piece::Pawn && mv.from.file() != target.file() {
        captured = piece_val(Piece::Pawn);
    }

    let mut gain = [0i32; 32];
    gain[0] = captured;
    let mut occ = board.occupied() ^ mv.from.bitboard();
    let mut next_victim = attacker;
    let mut stm = opp(board.side_to_move());
    let mut d = 0usize;

    loop {
        let atk = attackers_to(board, target, occ) & board.colors(stm);
        let (sq, p) = match least_valuable(board, atk) {
            Some(x) => x,
            None => break,
        };
        // a king cannot capture if the opponent still attacks the square afterwards
        if p == Piece::King {
            let after = attackers_to(board, target, occ ^ sq.bitboard()) & board.colors(opp(stm));
            if !after.is_empty() {
                break;
            }
        }
        d += 1;
        if d >= 32 {
            break;
        }
        gain[d] = piece_val(next_victim) - gain[d - 1];
        if (-gain[d - 1]).max(gain[d]) < 0 {
            break; // neither side can improve — prune the swap
        }
        occ ^= sq.bitboard();
        next_victim = p;
        stm = opp(stm);
    }
    while d > 0 {
        gain[d - 1] = -((-gain[d - 1]).max(gain[d]));
        d -= 1;
    }
    gain[0]
}

fn build_lmr_table() -> [[i8; 64]; 64] {
    let base = tune::lmr_base() as f64 / 100.0;
    let div = tune::lmr_div() as f64 / 100.0;
    let mut t = [[0i8; 64]; 64];
    for (d, row) in t.iter_mut().enumerate().skip(1) {
        for (m, r) in row.iter_mut().enumerate().skip(1) {
            *r = (base + (d as f64).ln() * (m as f64).ln() / div) as i8;
        }
    }
    t
}

// 2026-08-18 FIX: the LMR table now lives in the Searcher (field `lmr`) and is rebuilt once per
// search in think(). It used to be a process-wide static handed out as a `&'static` reference:
// in the `tune` build that reference pointed into a Box which a later parameter change REPLACED
// under the mutex, freeing the allocation while readers still held the reference -- a
// use-after-free. The old comment claimed "every reader takes an immutable snapshot", but
// replacing a Box deallocates it; there was no snapshot.
//
// Per-Searcher storage is also what the old comment said it wanted ("rebuilt per search from
// think(), not per node"), needs no unsafe, no mutex and no cfg split, and gives each SMP
// thread its own copy. Values are unchanged in the non-tune build, so this is behaviour-neutral
// on its own.

#[derive(Clone, Default)]
pub struct Limits {
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub movetime: Option<u128>,
    pub wtime: Option<u128>,
    pub btime: Option<u128>,
    pub winc: Option<u128>,
    pub binc: Option<u128>,
    /// Moves remaining before the clock is replenished, for a "40 moves in 15 minutes" style
    /// control. `None` means sudden death or increment-only.
    pub movestogo: Option<u32>,
    pub infinite: bool,
    /// Milliseconds reserved per move for GUI/network latency, subtracted from the clock before
    /// the budget is computed (UCI `Move Overhead`).
    pub move_overhead: u128,
    /// Experimental soft-budget adjustment from completed root iterations.
    pub adaptive_time: bool,
}

pub struct Searcher {
    pub tt: Arc<TT>,
    // true for exactly one thread per `go` (the one that prints `info`/returns bestmove and
    // bumps the TT age once); SMP helper threads share the same tt+stop but are not main.
    is_main: bool,
    /// Which thread this is in the SMP group. Used only for Lazy SMP search diversity; thread
    /// 0 (main, and every single-threaded caller) starts at depth 1, so Threads=1 is identical.
    pub thread_id: usize,
    killers: [[u16; 2]; MAX_PLY],
    // i16, not i32, 2026-09-11. These three tables are the hottest RANDOM-access data in the
    // engine -- `cont` alone is 1152 KB live against a 256 KB L2, so every continuation lookup is
    // an L3 hit, twice per quiet, ~33 quiets a node. Halving them halves the cache lines touched.
    //
    // BIT-EXACT, because the values provably fit. The gravity update
    // `e += bonus - e*|bonus|/16384` has a fixed point at exactly 16384 and never overshoots it:
    // at e=16384 with the largest bonus SPSA can ask for (HistBonusMax = 8000) the result is
    // 16384 again. So |value| <= 16384, comfortably inside i16. The ARITHMETIC still widens to
    // i32, because e*bonus reaches 131M.
    history: [[[i16; 64]; 64]; 2],
    counter: [[[u16; 64]; 64]; 2], // [stm][prev.from][prev.to] -> packed countermove
    // continuation history: two offset blocks (1-ply-ago, 2-ply-ago), each indexed
    // [prev_piece*64+prev_to][cur_piece*64+cur_to]; flat to avoid a large on-stack array.
    cont: Vec<i16>,
    cont_stack: Vec<usize>, // (piece,to) index of each move on the path; NULL_CONT for null moves
    // static eval per ply, so a node can ask whether the side to move is better off than it
    // was two plies ago ("improving"). Used to widen the late-move-pruning schedule.
    eval_hist: [i32; MAX_PLY],
    // move excluded at each ply during a singular-verification search (0 = none)
    excluded: [u16; MAX_PLY],
    // F5: the loaded network, resolved ONCE per search instead of per move. `nnue::net()` is a
    // OnceLock::get -- an atomic load plus a branch -- and it sat on the accumulator-update path
    // (once per make) and the eval path (once per node).
    net: Option<&'static nnue::Network>,
    // LMR reduction table, rebuilt once per search in think() (see build_lmr_table)
    lmr: [[i8; 64]; 64],
    // Ply at which null-move pruning is currently suppressed (-1 = nowhere). Set only around a
    // verification re-search, which re-enters negamax at the SAME ply on the SAME position and
    // would otherwise recurse into its own null move forever.
    nmp_off_ply: i32,
    // Depth of the iterative-deepening iteration in progress, for the check-extension cap.
    root_depth: i32,
    // Best move found at the root by the last completed iteration, packed. Tracked directly
    // rather than re-probed from the TT, which could have been evicted mid-search.
    root_best: u16,
    // capture history: [stm][moving piece*64 + to][victim] -> learned "how often did this
    // capture actually cut". MVV-LVA and SEE both judge a capture by the material on the
    // board; this judges it by what the search has already learned about it.
    capt: Vec<i16>,
    // captures searched at this node without cutting, so a fail-high can penalise them --
    // the capture-side mirror of `quiets_tried`.
    capt_buf: Vec<Vec<(Move, u8)>>,
    // ProbCut candidate captures. Its own buffer rather than a borrowed one: the probcut
    // search recurses, and reusing a buffer that the recursion also touches is how a subtle
    // aliasing bug gets in.
    pc_buf: Vec<Vec<(Move, Piece)>>,
    // correction history: [stm][pawn-structure bucket] -> learned static-eval offset, in
    // 1/CORR_GRAIN pawn units. Heap-allocated: 2*16384 i32 is 128 KB, too big for the stack.
    corr: Vec<i32>,
    // Per-ply move-ordering scratch. These used to be `Vec::with_capacity(48)` and
    // `Vec::new()` created fresh at every node -- a malloc/free pair per node at ~1.7M
    // nodes/s, and `quiets_tried` had no capacity at all so it reallocated as it grew.
    // Owned by the Searcher and reused: the buffers keep their capacity for the life of
    // the search. Taken out with mem::take while in use so `self` stays borrowable for
    // the recursive call, and put back on the paths that matter.
    move_buf: Vec<Vec<(i32, Move, u8)>>,
    qmove_buf: Vec<Vec<(i32, Move, u8)>>,
    // N2 (2026-09-11): was (Move, usize) at 16 B/entry. `cur_ci` is at most 5*64+63 = 383, so
    // u16 is ample and the entry drops to 6 B. These lists are walked at every quiet cutoff --
    // LMP-capped at ~67 entries by depth 8 -- so 1 KB of touched memory becomes 400 B at the
    // moment the search is stall-bound.
    quiet_buf: Vec<Vec<(Move, u16)>>,
    pub nodes: u64,
    seldepth: i32,
    start: Instant,
    soft_ms: u128,
    hard_ms: u128,
    max_nodes: u64,
    stop: Arc<AtomicBool>,
    stopped: bool,
    path: Vec<u64>,
    pub game_hist: Vec<u64>,
    /// This side's own root scores across the game, oldest first (volatility time management)
    pub eval_hist_game: Vec<i32>,
    pub silent: bool,
    pub last_score: i32, // score of the last completed ID iteration (stm perspective)
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Self {
        Self::for_thread(Arc::new(TT::new(hash_mb)), Arc::new(AtomicBool::new(false)), true)
    }

    /// Build a Searcher sharing an existing TT + stop flag with other threads (Lazy SMP).
    /// Exactly one thread in a group should be `is_main` -- it alone bumps the TT age and
    /// prints `info`/returns the bestmove; helpers search silently to populate the shared TT.
    pub fn for_thread(tt: Arc<TT>, stop: Arc<AtomicBool>, is_main: bool) -> Self {
        Searcher {
            tt,
            is_main,
            thread_id: 0,
            killers: [[0; 2]; MAX_PLY],
            history: [[[0i16; 64]; 64]; 2],
            counter: [[[0; 64]; 64]; 2],
            cont: vec![0i16; CONT_BLOCKS * CONT_STRIDE],
            corr: vec![0i32; 2 * CORR_SIZE],
            capt: vec![0i16; CAPT_SIZE],
            capt_buf: (0..MAX_PLY + 4).map(|_| Vec::with_capacity(32)).collect(),
            pc_buf: (0..MAX_PLY + 4).map(|_| Vec::with_capacity(32)).collect(),
            move_buf: (0..MAX_PLY + 4).map(|_| Vec::with_capacity(64)).collect(),
            qmove_buf: (0..MAX_PLY + 4).map(|_| Vec::with_capacity(32)).collect(),
            quiet_buf: (0..MAX_PLY + 4).map(|_| Vec::with_capacity(64)).collect(),
            cont_stack: Vec::with_capacity(MAX_PLY + 4),
            eval_hist: [0; MAX_PLY],
            excluded: [0; MAX_PLY],
            net: None,
            lmr: [[0i8; 64]; 64],
            root_best: 0,
            nmp_off_ply: -1,
            root_depth: 0,
            nodes: 0,
            seldepth: 0,
            start: Instant::now(),
            soft_ms: u128::MAX,
            hard_ms: u128::MAX,
            max_nodes: u64::MAX,
            stop,
            stopped: false,
            path: Vec::with_capacity(MAX_PLY + 4),
            game_hist: Vec::new(),
            eval_hist_game: Vec::new(),
            silent: false,
            last_score: 0,
        }
    }

    fn check_stop(&mut self) -> bool {
        if self.stopped {
            return true;
        }
        if self.nodes % 2048 == 0
            && (self.stop.load(Ordering::Relaxed)
                || self.nodes >= self.max_nodes
                || self.start.elapsed().as_millis() > self.hard_ms)
        {
            self.stopped = true;
        }
        self.stopped
    }

    /// Has `hash` occurred before, within reach of a repetition?
    ///
    /// 2026-09-10: this used to scan the WHOLE of `game_hist` with `.any()` at every node. A
    /// BENCH-INVISIBLE COST: bench constructs each position fresh so `game_hist` is always empty,
    /// and the engine's primary instrument therefore cannot see this at all. Measured under real
    /// conditions instead -- same position, 4,000,000 nodes pinned, reached via FEN (no history)
    /// versus via 90 moves -- it was **+5.3% slower per node** with a 90-ply history, and the cost
    /// grows linearly with move number. CCRL 40/15 games routinely run past 120 plies.
    ///
    /// Two bounds, both exact rather than heuristic:
    ///
    ///  * A repetition cannot reach past the last irreversible move, so only the most recent
    ///    `halfmove_clock` plies can possibly match. Anything older differs in material or pawn
    ///    structure and so differs in hash.
    ///  * Only EVEN plies back can match, because the side to move is part of the hash (verified:
    ///    the same placement with white and with black to move hashes differently). The immediate
    ///    parent has the opposite mover, so the scan starts two plies back and steps by two.
    ///
    /// Both only remove comparisons that could not have matched, so the tree is unchanged -- the
    /// bench signature is the gate on that claim.
    fn is_repetition(&self, hash: u64, halfmove: u8) -> bool {
        let budget = halfmove as usize;
        if budget < 2 {
            return false; // the last move was irreversible: nothing can repeat yet
        }
        // `path` holds ancestors of this node, most recent last.
        let from_path = self.path.len().min(budget);
        if self
            .path
            .iter()
            .rev()
            .take(from_path)
            .skip(1)
            .step_by(2)
            .any(|&h| h == hash)
        {
            return true;
        }
        // Whatever budget is left reaches back into the pre-search game history. Parity carries
        // over: `path` consumed `from_path` plies, so the first game_hist entry to test is the
        // one that keeps the two-ply stride.
        let rem = budget.saturating_sub(from_path);
        if rem == 0 {
            return false;
        }
        let skip = if from_path % 2 == 0 { 1 } else { 0 };
        self.game_hist
            .iter()
            .rev()
            .take(rem)
            .skip(skip)
            .step_by(2)
            .any(|&h| h == hash)
    }

    /// static eval at a node: NNUE (incremental accumulator) when a net is loaded, else PeSTO.
    /// Invariant: `acc` is Some iff a net is loaded (established at the root in think()).
    #[inline]
    fn eval_node(&self, board: &Board, acc: Option<&nnue::Acc>) -> i32 {
        // 2026-09-11 (readiness audit R2): K+B vs K measured score cp 160 -- the engine
        // believed it was better by 1.6 pawns in a position it cannot win, and played on,
        // spending clock. The popcount gate means every real position pays one `len()` and a
        // predictable not-taken branch.
        if insufficient_material(board) {
            return 0;
        }
        match acc {
            Some(a) => nnue::forward(self.net.unwrap(), &a.w, &a.b, board.side_to_move()),
            None => evaluate(board),
        }
    }

    /// Bucket for correction history: the pawn structure, and nothing else.
    ///
    /// The two pawn bitboards DETERMINE the structure exactly, so they are hashed directly
    /// rather than walked square by square -- this is O(1), which matters at 1.7 M nodes/s.
    /// (A Zobrist-style incremental key would need threading through make/unmake; the board
    /// here is rebuilt per node, so there is nothing to thread.) splitmix64 finalizer, because
    /// raw bitboards have terrible low-bit entropy and the index is a mask of the low bits.
    fn corr_index(board: &Board, stm: Color) -> usize {
        let p = board.pieces(Piece::Pawn);
        let w = (p & board.colors(Color::White)).0;
        let b = (p & board.colors(Color::Black)).0;
        let mut x = w
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ b.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 31;
        x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
        x ^= x >> 29;
        (stm as usize) * CORR_SIZE + (x as usize & (CORR_SIZE - 1))
    }

    /// Static eval plus whatever the search has learned about this pawn structure.
    fn corrected_eval(&self, board: &Board, stm: Color, raw: i32) -> i32 {
        if !CORR_HIST {
            return raw;
        }
        self.corrected_eval_at(Self::corr_index(board, stm), raw)
    }

    /// F6 (2026-09-10): the index form. `corr_index` is a two-multiply splitmix hash of both pawn
    /// bitboards, and it was computed TWICE per node -- once here and again in `update_corr` at
    /// the end of the same node, for the same position. Hash once, pass the index down.
    #[inline]
    fn corrected_eval_at(&self, ci: usize, raw: i32) -> i32 {
        if !CORR_HIST {
            return raw;
        }
        raw + self.corr[ci] / CORR_GRAIN
    }

    /// Fold one search result into the correction for this structure.
    ///
    /// Guards, all of which matter:
    ///  * not in check -- the static eval is meaningless there, so the difference is noise;
    ///  * best move is not a capture -- that difference is material the eval already saw
    ///    change, not a standing bias in how this structure is judged;
    ///  * the bound has to point the right way. A fail-high only proves `best` is a LOWER
    ///    bound, so it is evidence the eval was too low and evidence of nothing at all if the
    ///    eval was already higher. Symmetrically for fail-low. Without this the table learns
    ///    from bounds that never constrained anything.
    fn update_corr(
        &mut self,
        corr_ci: usize,
        static_eval: i32,
        best: i32,
        bound: u8,
        depth: i32,
        best_is_capture: bool,
    ) {
        if !CORR_HIST || best_is_capture || best.abs() >= MATE_BOUND {
            return;
        }
        if (bound == BOUND_LOWER && best <= static_eval)
            || (bound == BOUND_UPPER && best >= static_eval)
        {
            return;
        }
        let w = depth.clamp(1, CORR_W_MAX);
        let target = (best - static_eval).clamp(-CORR_MAX / CORR_GRAIN, CORR_MAX / CORR_GRAIN)
            * CORR_GRAIN;
        let e = &mut self.corr[corr_ci];
        *e = (*e * (CORR_W - w) + target * w) / CORR_W;
        *e = (*e).clamp(-CORR_MAX, CORR_MAX);
    }

    // ---------------- quiescence ----------------
    fn qsearch(
        &mut self,
        board: &Board,
        ply: i32,
        mut alpha: i32,
        beta: i32,
        acc: Option<&nnue::Acc>,
    ) -> i32 {
        self.nodes += 1;
        self.seldepth = self.seldepth.max(ply);
        if self.check_stop() || ply as usize >= MAX_PLY - 1 {
            return self.eval_node(board, acc);
        }
        let stm = board.side_to_move();
        let in_check = !board.checkers().is_empty();

        // Transposition probe + store (2026-09-09). qsearch had neither -- the single largest
        // standard omission in this search: quiescence is ~63% of all nodes (216481 of 343299
        // at bench) and every one re-derived a result the table may already hold. Worth a third
        // of the whole tree: bench 673771 -> 448187.
        //
        // THE DEPTH TEST IS LOAD-BEARING, and cost a gate failure to find. qsearch may cut only
        // on entries QSEARCH stored (depth 0). Returning a NEGAMAX-stored entry makes the search
        // report mate 2 at depth 5 and then mate 3 at depth 6 on the `mates` suite's rook-roller
        // -- a score getting WORSE with depth, which is a broken search, not a tuning artefact.
        // Leave-one-out over the entry space, all at bench depth 11:
        //
        //     probe qsearch entries only (d == 0)   bench 448187   mates PASS
        //     probe negamax entries only (d >= 1)   bench 541446   mates FAIL
        //     probe everything                      bench 507406   mates FAIL
        //     probe everything except EXACT bounds  bench 551139   mates FAIL
        //
        // so it is negamax-stored entries specifically, and NOT the EXACT bound -- excluding
        // those alone does not rescue it. THE MECHANISM IS UNCONFIRMED; what is established is
        // the classification above, which is what this guard is written against. The plausible
        // reading is that a negamax entry describes a full-width search of the position while
        // qsearch's caller asked for the quiescence value, and negamax only ever consumes these
        // under `e.depth >= depth` -- a comparison qsearch has no depth to make.
        //
        // The MOVE hint is taken from any entry at any depth: it is only an ordering
        // suggestion, is checked against generated moves, and cannot return a score.
        let hash = board.hash();
        let mut tt_mv: u16 = 0;
        if QS_TT {
            if let Some(e) = self.tt.probe(hash) {
                tt_mv = e.mv;
                if e.depth as i32 <= 0 {
                    let sc = tt_score_from(e.score as i32, ply);
                    match e.bound {
                        BOUND_EXACT => return sc,
                        BOUND_LOWER if sc >= beta => return sc,
                        BOUND_UPPER if sc <= alpha => return sc,
                        _ => {}
                    }
                }
            }
        }

        let alpha_orig = alpha;
        let mut best_mv: u16 = 0;
        let mut best;
        if in_check {
            best = -INF; // must search all evasions; no stand-pat while in check
        } else {
            let raw = self.eval_node(board, acc);
            best = if CORR_QSEARCH {
                self.corrected_eval(board, stm, raw)
            } else {
                raw
            };
            if best >= beta {
                return best;
            }
            alpha = alpha.max(best);
        }

        let mut moves: Vec<(i32, Move, u8)> = std::mem::take(&mut self.qmove_buf[ply as usize]);
        moves.clear();
        // Destination mask (2026-09-09). qsearch used to walk EVERY legal move and throw away
        // the ~85% that are quiet -- and the throwing-away is not free: each rejected move costs
        // a Move construction plus the `capture_victim` board lookups. cozy's `generate_moves_for`
        // masks source PIECES, not destinations, so the mask goes on `pm.to` instead.
        //
        // Exactness: a move outside this mask lands on an empty non-ep, non-promotion square, so
        // `victim` is None and `is_qpromo` false, and the filter below would have dropped it
        // anyway. Same move list, same order, same tree -- the bench signature is unchanged.
        let qtargets = if in_check {
            BitBoard::FULL // in check every evasion is searched, quiet or not
        } else {
            let mut t = board.colors(opp(stm));
            if let Some(f) = board.en_passant() {
                // the ep capture's destination is EMPTY, so it is not in the enemy occupancy
                t |= Square::new(f, Rank::Sixth.relative_to(stm)).bitboard();
            }
            t
        };
        let promo_rank = Rank::Eighth.relative_to(stm).bitboard();
        // F4: hoisted once per node; the classifier below needs no board queries of its own.
        let enemy = board.colors(opp(stm));
        let ep_sq = ep_square(board, stm);
        board.generate_moves(|mut pm| {
            if !in_check {
                let mut t = qtargets;
                if pm.piece == Piece::Pawn {
                    t |= promo_rank; // queen promotions are searched even when they capture nothing
                }
                pm.to &= t;
                if pm.to.is_empty() {
                    return false;
                }
            }
            let moved = pm.piece;
            for mv in pm {
                let victim = capture_victim_with(board, mv, enemy, ep_sq);
                let is_qpromo = mv.promotion == Some(Piece::Queen);
                if in_check || victim.is_some() || is_qpromo {
                    // SEE pruning: skip losing captures entirely (not while in check)
                    if !in_check && victim.is_some() && !is_qpromo && see_with(board, mv, moved) < 0
                    {
                        continue;
                    }
                    let attacker = moved;
                    let mut s = 0;
                    if QS_TT && tt_mv != 0 && pack(mv) == tt_mv {
                        s = 1_000_000; // the table's move first, as in negamax
                    } else {
                        if let Some(v) = victim {
                            s += 100_000 + 10 * piece_val(v) - piece_val(attacker);
                        }
                        if is_qpromo {
                            s += 90_000;
                        }
                    }
                    // The tag carries BOTH the mover (bits 0..2) and the victim+1 (bits 3..5),
                    // so neither the delta-pruning pass nor the accumulator update below needs a
                    // board query. Still one u8: (i32, Move, u8) is 8 bytes, (i32, Move, u16)
                    // would be 12 because Move is 3 bytes, not 2.
                    let vt = moved as u8 | (victim.map_or(0u8, |v| v as u8 + 1) << 3);
                    moves.push((s, mv, vt));
                }
            }
            false
        });

        if in_check && moves.is_empty() {
            self.qmove_buf[ply as usize] = moves;
            return -MATE + ply;
        }
        moves.sort_unstable_by_key(|&(s, _, _)| -s);

        for &(_, mv, vtag) in moves.iter() {
            let qmoved = Piece::index((vtag & 7) as usize);
            let qvic = vtag >> 3;
            // delta pruning: even winning this victim can't lift alpha
            if !in_check && qvic != 0 {
                // victim recovered from the tag set during generation, not reclassified.
                let v = Piece::index((qvic - 1) as usize);
                if best + piece_val(v) + 200 < alpha {
                    continue;
                }
            }
            let mut nb = board.clone();
            nb.play_unchecked(mv);
            self.tt.prefetch(nb.hash());
            let qvictim = if qvic != 0 {
                let p = Piece::index((qvic - 1) as usize);
                let sq = if Some(mv.to) == ep_sq {
                    Square::new(mv.to.file(), mv.from.rank())
                } else {
                    mv.to
                };
                Some((p, sq))
            } else {
                None
            };
            let nacc = acc.map(|a| {
                nnue::acc_update_known(self.net.unwrap(), a, board, mv, qmoved, qvictim)
            });
            let sc = -self.qsearch(&nb, ply + 1, -beta, -alpha, nacc.as_ref());
            if self.stopped {
                self.qmove_buf[ply as usize] = moves;
                return best.max(sc);
            }
            if sc > best {
                best = sc;
                best_mv = pack(mv);
                if sc > alpha {
                    alpha = sc;
                    if sc >= beta {
                        break;
                    }
                }
            }
        }
        self.qmove_buf[ply as usize] = moves;
        if QS_TT && !self.stopped {
            // Bound is decided the same way negamax decides it: a score that never beat the
            // original alpha is an upper bound, one that reached beta is a lower bound.
            let bound = if best >= beta {
                BOUND_LOWER
            } else if best > alpha_orig {
                BOUND_EXACT
            } else {
                BOUND_UPPER
            };
            self.tt.store(hash, best_mv, tt_score_to(best, ply), 0, bound);
        }
        best
    }

    // continuation-history update with the same gravity as butterfly history. `bonus` is signed:
    // positive rewards, negative penalizes. Applied to whichever predecessor slots exist.
    #[inline]
    fn cont_bonus(&mut self, preds: &[usize; CONT_BLOCKS], ci: usize, bonus: i32) {
        for b in 0..cont_blocks() {
            // The sentinel (CONT_PT) must not be written, or the zero row would stop being zero.
            if preds[b] != CONT_PT {
                let e = &mut self.cont[b * CONT_STRIDE + preds[b] * CONT_PT + ci];
                let v = *e as i32;
                *e = (v + bonus - v * bonus.abs() / 16_384) as i16;
            }
        }
    }

    /// The (piece,to) index of the move played 1, 2, 4 and 6 plies ago; `CONT_PT` when absent.
    #[inline]
    fn cont_preds(&self) -> [usize; CONT_BLOCKS] {
        let n = self.cont_stack.len();
        let mut out = [CONT_PT; CONT_BLOCKS]; // CONT_PT = the zero sentinel row
        for (b, &back) in CONT_PLIES.iter().enumerate().take(cont_blocks()) {
            if n >= back {
                let p = self.cont_stack[n - back];
                if p != NULL_CONT {
                    out[b] = p;
                }
            }
        }
        out
    }

    /// Sum of the continuation-history entries for `ci` under the current predecessors.
    #[inline]
    fn cont_score(&self, preds: &[usize; CONT_BLOCKS], ci: usize) -> i32 {
        let mut h = 0;
        for b in 0..cont_blocks() {
            h += self.cont[b * CONT_STRIDE + preds[b] * CONT_PT + ci] as i32
                / CONT_WEIGHT_DIV[b];
        }
        h
    }

    // ---------------- main search ----------------
    #[allow(clippy::too_many_arguments)]
    fn negamax(
        &mut self,
        board: &Board,
        mut depth: i32,
        ply: i32,
        mut alpha: i32,
        beta: i32,
        prev: u16,
        acc: Option<&nnue::Acc>,
        cut_node: bool,
    ) -> i32 {
        if self.check_stop() {
            return 0;
        }
        let hash = board.hash();
        if ply > 0 {
            if board.halfmove_clock() >= 100 || self.is_repetition(hash, board.halfmove_clock()) {
                self.nodes += 1;
                return 0;
            }
        }

        let in_check = !board.checkers().is_empty();
        if in_check {
            // Check extension. Uncapped, a forcing sequence can extend a line indefinitely --
            // harmless at bench depth 11, much less so at the ~depth 22-25 a 40/15 search
            // reaches. The cap lets checks extend freely until the line is already twice the
            // root depth, which is where a genuine forcing line has long since resolved.
            if !CHECK_EXT_CAP || ply < 2 * self.root_depth {
                depth += 1;
            }
        }
        if depth <= 0 {
            return self.qsearch(board, ply, alpha, beta, acc);
        }
        self.nodes += 1;
        if ply as usize >= MAX_PLY - 1 {
            return self.eval_node(board, acc);
        }

        // Mate-distance pruning. A mate found at this ply cannot be better than MATE - ply, nor
        // worse than -MATE + ply, so a window already outside that range has nothing to find.
        // Costs two compares and mainly buys shorter mate lines in won positions.
        if MATE_DIST && ply > 0 {
            let a = alpha.max(-MATE + ply);
            let b = beta.min(MATE - ply - 1);
            if a >= b {
                return a;
            }
            alpha = a;
        }

        // TT probe
        let mut tt_mv: u16 = 0;
        // A move excluded at this node means we are inside a singular-verification search: the
        // position is being searched WITHOUT its best move, so it is not the same search problem
        // as the plain position. Probing or storing under `hash` would cross-contaminate the two,
        // so both are skipped for the duration.
        let excl = self.excluded[ply as usize];

        let mut tt_depth: i32 = -1;
        let mut tt_score: i32 = 0;
        let mut tt_bound: u8 = 0;
        // Hoisted above the probe (2026-08-18). NOT used by the TT cutoff below -- it feeds
        // LMP/LMR later. It only reads the window, so it is computable this early.
        let is_pv = beta - alpha > 1;
        if excl == 0 {
            if let Some(e) = self.tt.probe(hash) {
                tt_mv = e.mv;
                tt_depth = e.depth as i32;
                tt_score = tt_score_from(e.score as i32, ply);
                tt_bound = e.bound;
                // DELIBERATELY NO PV GUARD. An earlier comment here claimed "no TT cutoff in a
                // PV node"; this code has never had that guard. Adding it was measured at
                // -38.21 +/- 18.34 and removed (c34ad12). The cutoff below is depth- and
                // bound-gated only -- do not re-add a PV exclusion without an SPRT.
                if ply > 0 && e.depth as i32 >= depth {
                    let sc = tt_score_from(e.score as i32, ply);
                    match e.bound {
                        BOUND_EXACT => return sc,
                        BOUND_LOWER if sc >= beta => return sc,
                        BOUND_UPPER if sc <= alpha => return sc,
                        _ => {}
                    }
                }
            }
        }

        // Internal iterative reduction: with no TT move the ordering here is only history/policy,
        // so a full-depth search is poor value -- searching one ply shallower fills the TT with a
        // best move, and the next visit to this node gets it properly ordered. Cheaper than a real
        // internal iterative deepening re-search.
        if IIR_MIN_DEPTH > 0 && depth >= IIR_MIN_DEPTH && tt_mv == 0 {
            depth -= 1;
            // With no table move the ordering here is guesswork, and at a node we already
            // expect to fail high that is a bad place to spend depth. Take a second ply.
            if CUT_NODE && cut_node && depth >= CUT_IIR_MIN_DEPTH {
                depth -= 1;
            }
        }

        let stm = board.side_to_move();
        let raw_eval = self.eval_node(board, acc);
        // Everything downstream -- improving, reverse futility, null move, LMP -- reads the
        // CORRECTED eval. That is the point: the corrections are only worth anything if they
        // reach the pruning decisions the static eval drives.
        // F6: hashed once here, reused by update_corr at the end of this node.
        let corr_ci = if CORR_HIST {
            Self::corr_index(board, stm)
        } else {
            0
        };
        let static_eval = self.corrected_eval_at(corr_ci, raw_eval);
        self.eval_hist[ply as usize] = static_eval;

        // "improving": the side to move stands better than it did two plies ago, so this line is
        // going our way and late quiets deserve a longer look before being pruned. Meaningless
        // while in check (the static eval of a check position says nothing), so force it false.
        let improving = !in_check
            && ply >= 2
            && static_eval > self.eval_hist[ply as usize - 2];

        // reverse futility pruning
        let rfp_depth = if RFP_IMPROVING { depth - improving as i32 } else { depth };
        if !in_check && ply > 0 && depth <= 6 && static_eval - tune::rfp_margin() * rfp_depth >= beta
        {
            return static_eval;
        }

        // Razoring. When the static eval is this far below alpha with little depth left, the
        // node is very unlikely to reach alpha by quiet play. Rather than trust that outright,
        // verify with a quiescence search at the same window: only a qsearch that ALSO fails
        // low returns. That verification is what separates razoring from simply forfeiting the
        // node, and it is cheap because qsearch is where the position was heading anyway.
        if RAZOR
            && !is_pv
            && !in_check
            && ply > 0
            && depth <= RAZOR_MAX_DEPTH
            && static_eval + RAZOR_MARGIN * depth < alpha
        {
            let s = self.qsearch(board, ply, alpha - 1, alpha, acc);
            if self.stopped {
                return 0;
            }
            if s < alpha {
                return s;
            }
        }

        // null-move pruning
        if !in_check
            && ply > 0
            && depth >= 3
            && static_eval >= beta
            && has_non_pawn(board, stm)
            && self.nmp_off_ply != ply
        {
            if let Some(nb) = board.null_move() {
                let mut r = tune::nmp_base() + depth / tune::nmp_div();
                if NMP_EVAL_R {
                    r += ((static_eval - beta) / 200).clamp(0, 3);
                }
                self.path.push(hash);
                self.cont_stack.push(NULL_CONT);
                // null move: no pieces change, accumulator carries over unchanged
                let sc = -self.negamax(&nb, depth - 1 - r, ply + 1, -beta, -beta + 1, 0, acc, !cut_node);
                self.cont_stack.pop();
                self.path.pop();
                if self.stopped {
                    return 0;
                }
                if sc >= beta {
                    // Verification. A null-move cutoff asserts "the position is so good that
                    // even passing beats beta" -- which is exactly false in zugzwang, where
                    // passing is the best move available and every real move loses ground.
                    // has_non_pawn() screens the crude endgame case; it does not screen a
                    // middlegame squeeze. Above NMP_VERIFY_DEPTH the cutoff is re-searched at
                    // reduced depth with null disabled AT THIS PLY (deeper plies keep it), and
                    // only a second fail-high returns. Deep nodes are rare and expensive to get
                    // wrong, which is why the check is bought only there.
                    let verified = if NMP_VERIFY && depth >= NMP_VERIFY_DEPTH {
                        let saved = self.nmp_off_ply;
                        self.nmp_off_ply = ply;
                        let v = self.negamax(board, depth - r, ply, beta - 1, beta, prev, acc, cut_node);
                        self.nmp_off_ply = saved;
                        if self.stopped {
                            return 0;
                        }
                        v >= beta
                    } else {
                        true
                    };
                    // A failed verification does NOT return -- it falls through to the ordinary
                    // move loop below, which is the whole point: the node gets searched for
                    // real instead of being written off on the strength of a pass.
                    if verified {
                        // Kept fail-hard DELIBERATELY. Returning `sc` is the textbook fail-soft
                        // form, but the null score comes from a REDUCED-depth search in which
                        // the opponent was handed a free move, and propagating it measured
                        // negative here (2026-08-21). `beta` bounds that damage.
                        return beta;
                    }
                }
            }
        }

        // ProbCut. A capture that clears beta + PROBCUT_MARGIN at reduced depth is very unlikely
        // to fail to clear beta at full depth, so the node can be cut without the full search.
        //
        // Order of work matters for cost: candidates are screened by SEE against the margin they
        // have to cover, then by a QSEARCH at the raised window, and only survivors of both pay
        // for a reduced-depth negamax. Skipped when the table already says this node cannot
        // reach the raised bound at comparable depth -- that is the cheapest screen of all.
        if PROBCUT
            && !is_pv
            && !in_check
            && ply > 0
            && depth >= PROBCUT_MIN_DEPTH
            && beta.abs() < MATE_BOUND
            && !(tt_depth >= depth - PROBCUT_REDUCTION && tt_score < beta + PROBCUT_MARGIN
                 && tt_bound != 0u8)
        {
            let pc_beta = beta + PROBCUT_MARGIN;
            let pc_depth = depth - PROBCUT_REDUCTION;
            let mut cands: Vec<(Move, Piece)> = std::mem::take(&mut self.pc_buf[ply as usize]);
            cands.clear();
            let enemy = board.colors(opp(stm));
            board.generate_moves(|mut pm| {
                pm.to &= enemy;
                let pcp = pm.piece;
                for mv in pm {
                    if see_with(board, mv, pcp) >= pc_beta - static_eval {
                        cands.push((mv, pcp));
                    }
                }
                false
            });
            for &(mv, pcp) in cands.iter() {
                let mut nb = board.clone();
                nb.play_unchecked(mv);
                self.tt.prefetch(nb.hash());
                let pvictim = board.piece_on(mv.to).map(|v| (v, mv.to));
                let nacc = acc.map(|a| {
                    nnue::acc_update_known(self.net.unwrap(), a, board, mv, pcp, pvictim)
                });
                let na = nacc.as_ref();
                let cur_ci = pcp as usize * 64 + mv.to as usize;
                self.path.push(hash);
                self.cont_stack.push(cur_ci);
                // cheap screen first
                let mut v = -self.qsearch(&nb, ply + 1, -pc_beta, -pc_beta + 1, na);
                if v >= pc_beta && pc_depth >= 1 {
                    v = -self.negamax(&nb, pc_depth, ply + 1, -pc_beta, -pc_beta + 1, pack(mv), na, !cut_node);
                }
                self.cont_stack.pop();
                self.path.pop();
                if self.stopped {
                    self.pc_buf[ply as usize] = cands;
                    return 0;
                }
                if v >= pc_beta {
                    self.pc_buf[ply as usize] = cands;
                    return v;
                }
            }
            self.pc_buf[ply as usize] = cands;
        }

        // continuation-history predecessors for this node: 1-ply-ago (counter-move history) and
        // 2-ply-ago (follow-up history). cont_stack.len() == ply here (each ply pushes once).
        let preds = self.cont_preds();

        // generate + score. Arm A (Stage 3): when a policy net is loaded, QUIET moves are
        // ranked by the policy logit instead of killers/countermove/history — captures,
        // promotions and the TT move keep their classical slots. act512 computed once per
        // node from the same accumulator the value head uses.
        // F4: hoisted so the per-move classifier needs no board queries of its own.
        let enemy = board.colors(opp(stm));
        let ep_sq = ep_square(board, stm);
        let pol = crate::policy::policy();
        let act = match (pol, acc) {
            (Some(_), Some(a)) => Some(crate::policy::activations(a, stm)),
            _ => None,
        };
        let mut moves: Vec<(i32, Move, u8)> = std::mem::take(&mut self.move_buf[ply as usize]);
        moves.clear();
        board.generate_moves(|pm| {
            // The generator already tells us which piece is moving. `board.piece_on(mv.from)`
            // rescans up to six bitboards to rediscover it, once per move, ~33 times a node.
            // Same value, no scan -- and the bench signature proves it is the same value.
            let moved = pm.piece;
            for mv in pm {
                let packed = pack(mv);
                let mut tg = moved as u8;
                // NB: must read the victim through capture_victim, not piece_on(mv.to) --
                // for en passant the destination is empty and .unwrap() would panic.
                let victim = capture_victim_with(board, mv, enemy, ep_sq);
                // CLASSIFICATION IS SET HERE, OUTSIDE THE SCORING CHAIN, and that placement is
                // the whole correctness argument. Setting it inside the capture arm lets the
                // TT-move arm short-circuit past it, so a TT move that IS a capture arrives in
                // the loop tagged quiet -- counted in quiets_tried, given killer/history
                // updates, made LMR-eligible. The bench signature caught exactly that: 271991
                // instead of 284537.
                if victim.is_some() {
                    tg |= tag::CAPTURE;
                }
                if let Some(v) = victim {
                    // victim+1 in bits 5..7, set OUTSIDE the scoring chain for the same reason
                    // CAPTURE is: a TT-move capture must still carry its victim, and the TT arm
                    // short-circuits past the capture scoring arm below.
                    tg |= (v as u8 + 1) << 5;
                }
                let s = if packed == tt_mv && tt_mv != 0 {
                    1_000_000
                } else if let Some(v) = victim {
                    let a = moved;
                    let mvvlva = 10 * piece_val(v) - piece_val(a);
                    // MVV-LVA and SEE both judge a capture by the material standing on the
                    // board. Capture history adds what the search has actually learned about
                    // this (piece, destination, victim) triple. Divided down so it re-ranks
                    // captures against each other without ever lifting a losing capture out of
                    // its bucket or above the TT move.
                    let ch = if CAPT_HIST {
                        self.capt[capt_index(stm, a, mv.to, v)] as i32 / 64
                    } else {
                        0
                    };
                    if see_with(board, mv, moved) >= 0 {
                        tg |= tag::SEE_WIN;
                        100_000 + mvvlva + ch // winning/equal captures ahead of all but TT
                    } else {
                        -20_000 + mvvlva / 10 + ch // losing captures behind all quiets
                    }
                } else if mv.promotion == Some(Piece::Queen) {
                    95_000
                } else if let (Some(p), Some(a)) = (pol, act.as_ref()) {
                    // policy ordering for quiets (incl. quiet underpromotions)
                    let lg = crate::policy::logit(p, a, crate::policy::move_class(board, mv));
                    50_000 + (lg >> 8).clamp(-30_000, 30_000)
                } else if packed == self.killers[ply as usize][0] {
                    80_000
                } else if packed == self.killers[ply as usize][1] {
                    79_999
                } else if prev != 0
                    && packed
                        == self.counter[stm as usize][(prev & 63) as usize]
                            [((prev >> 6) & 63) as usize]
                {
                    78_000
                } else {
                    // butterfly history + continuation history (1-ply + 2-ply predecessors)
                    tg |= tag::IS_HIST_SCORE;
                    let ci = moved as usize * 64 + mv.to as usize;
                    self.history[stm as usize][mv.from as usize][mv.to as usize] as i32
                        + self.cont_score(&preds, ci)
                };
                moves.push((s, mv, tg));
            }
            false
        });

        if moves.is_empty() {
            self.move_buf[ply as usize] = moves;
            return if in_check { -MATE + ply } else { 0 };
        }
        moves.sort_unstable_by_key(|&(s, _, _)| -s);

        if KILLER_CLEAR && (ply as usize) + 2 < MAX_PLY {
            self.killers[ply as usize + 2] = [0; 2];
        }

        let mut best = -INF;
        let mut best_mv: u16 = 0;
        let mut best_is_capture = false;
        let mut bound = BOUND_UPPER;
        let mut quiets_tried: Vec<(Move, u16)> =
            std::mem::take(&mut self.quiet_buf[ply as usize]);
        quiets_tried.clear();
        let mut caps_tried: Vec<(Move, u8)> = std::mem::take(&mut self.capt_buf[ply as usize]);
        caps_tried.clear();

        // Late-move-pruning schedule: how many quiets are worth searching at this depth before
        // the remainder -- which the ordering has already ranked worst -- are written off.
        // Doubled when improving, since a line that is going our way deserves the longer look.
        let lmp_limit = ((tune::lmp_base() + depth * depth) / if improving { 1 } else { 2 }) as usize;

        for (i, &(sc, mv, tg)) in moves.iter().enumerate() {
            // `sc` is shadowed below by the SEARCH score; keep the ordering score for the
            // cutoff instrument.
            #[cfg(feature = "instrument")]
            let ord_s = sc;
            let packed_mv = if excl != 0 { pack(mv) } else { u16::MAX };
            if packed_mv == excl {
                continue; // singular verification: this node is searched without its best move
            }
            // F1: was `capture_victim(board, mv).is_some()` -- a full reclassify per move,
            // including quiets the scoring closure had already proved quiet and moves about to be
            // pruned. cozy's piece_on is a linear scan of up to six bitboards.
            let is_capture = tag::is_capture(tg);
            let is_quiet = !is_capture && mv.promotion.is_none();

            // Late move pruning. Once `lmp_limit` quiets have been searched without beating
            // alpha, the rest are very unlikely to, so skip them -- before the board clone and
            // accumulator update, which is where the saving actually comes from. `quiets_tried`
            // counts only quiets that were *searched*, so once the limit is hit it stops growing
            // and every later quiet is pruned too. Never in a PV node, never in check, and never
            // while we are getting mated (there every defensive resource still has to be seen).
            if !is_pv
                && !in_check
                && is_quiet
                && depth <= LMP_MAX_DEPTH
                && best > -MATE_BOUND
                && quiets_tried.len() >= lmp_limit
            {
                continue;
            }

            // History pruning. A late quiet that the tables have specific evidence against is
            // worth less than the board clone it would cost. Distinct from LMP: that one counts
            // moves, this one reads what the search already learned about THIS move.
            if HIST_PRUNE
                && !is_pv
                && !in_check
                && is_quiet
                && i >= 3
                && depth <= HIST_PRUNE_MAX_DEPTH
                && best > -MATE_BOUND
            {
                let ci = board.piece_on(mv.from).unwrap() as usize * 64 + mv.to as usize;
                let h = self.history[stm as usize][mv.from as usize][mv.to as usize] as i32
                    + self.cont_score(&preds, ci);
                if h < HIST_PRUNE_THRESHOLD * depth {
                    continue;
                }
            }

            // Forward futility pruning. Near the horizon a quiet move that leaves the static
            // eval this far below alpha is very unlikely to lift it, so it is written off
            // before the board clone and accumulator update -- the same place, and for the
            // same reason, as LMP above.
            //
            // `best > -MATE_BOUND` is the "at least one move has been searched" guard: `best`
            // is -INF until the first move returns, and -INF is NOT > -MATE_BOUND. LMP relies
            // on exactly the same fact.
            if FUTILITY
                && !is_pv
                && !in_check
                && is_quiet
                && depth <= FUT_MAX_DEPTH
                && best > -MATE_BOUND
                && static_eval + FUT_MARGIN * depth <= alpha
            {
                continue;
            }

            // SEE pruning. A move that loses material outright by static exchange is worth
            // searching only if the depth left can plausibly justify it. Quiets are held to a
            // linear budget in depth, captures to a quadratic one -- a losing capture at least
            // wins material first, so it earns more rope.
            if SEE_PRUNE
                && !is_pv
                && !in_check
                && depth <= 8
                && best > -MATE_BOUND
                && mv.promotion.is_none()
            {
                // F2: the capture threshold is SEE_C_MARGIN * depth^2, always NEGATIVE, so a
                // capture the scoring closure already measured at see >= 0 can NEVER be pruned
                // here -- the call was pure waste of the most expensive function in the ordering
                // path. The tag carries that sign. Robust where score arithmetic would not be:
                // a TT move is searched first, where `best > -MATE_BOUND` is false and this
                // block cannot fire at all.
                if !(is_capture && tag::see_winning(tg)) {
                    let threshold = if is_quiet {
                        SEE_Q_MARGIN * depth
                    } else {
                        SEE_C_MARGIN * depth * depth
                    };
                    if see_with(board, mv, Piece::index(tag::piece(tg))) < threshold {
                        continue;
                    }
                }
            }

            // Singular extension. If the TT says this move is good enough to have caused a
            // fail-high at nearly this depth, ask whether it is the ONLY move that does: search
            // every other move at reduced depth against a window just below the TT score. If they
            // all fail low the move is singular -- the line hinges on it -- so it is worth an
            // extra ply. Requires `excl == 0`: inside a verification search the test must not
            // recurse, and the TT fields it reads were not probed there anyway.
            let mut extension = 0i32;
            if SE_MIN_DEPTH > 0
                && ply > 0
                && excl == 0
                && depth >= SE_MIN_DEPTH
                && pack(mv) == tt_mv
                && tt_mv != 0
                && tt_depth >= depth - 3
                && tt_bound != BOUND_UPPER
                && tt_score.abs() < MATE_BOUND
            {
                let s_beta = tt_score - (SE_MARGIN * depth) / 60;
                let s_depth = (depth - 1) / 2;
                self.excluded[ply as usize] = tt_mv;
                let s = self.negamax(board, s_depth, ply, s_beta - 1, s_beta, prev, acc, cut_node);
                self.excluded[ply as usize] = 0;
                if self.stopped {
                    self.move_buf[ply as usize] = moves;
                    self.quiet_buf[ply as usize] = quiets_tried;
                    self.capt_buf[ply as usize] = caps_tried;
                    return 0; // no move searched yet at this point, so nothing to preserve
                }
                if s < s_beta {
                    // Singular. Scale the extension by how far below the margin the excluded
                    // search fell: one ply, plus a second/third when it is far enough below.
                    extension = 1;
                    if SE_DOUBLE && s < s_beta - (SE_DOUBLE_MARGIN * depth) / 60 {
                        extension += 1;
                    }
                    if SE_TRIPLE && s < s_beta - (SE_TRIPLE_MARGIN * depth) / 60 {
                        extension += 1;
                    }
                } else if SE_MULTICUT && s >= beta && tt_score.abs() < MATE_BOUND {
                    // Multi-cut: the TT move was assumed to fail high, but other moves fail high
                    // over beta without it, so this expected cut node is not singular. Cut the
                    // whole subtree with a soft bound instead of searching it.
                    self.move_buf[ply as usize] = moves;
                    self.quiet_buf[ply as usize] = quiets_tried;
                    self.capt_buf[ply as usize] = caps_tried;
                    return s;
                } else if SE_NEGATIVE {
                    // Not singular and not a multi-cut. If the TT move is assumed to fail high
                    // over beta, reduce it hard; on a cut node reduce it a little. This is the
                    // "the stored move is not as good as the table thinks" signal.
                    extension = if tt_score >= beta { -3 } else if cut_node { -2 } else { 0 };
                }
            }
            let new_depth = depth - 1 + extension;

            // continuation-history index of the move being played (any move, capture or quiet)
            // F1: the generator handed us the piece; the buffer now carries it.
            let cur_ci = tag::piece(tg) * 64 + mv.to as usize;
            let mut nb = board.clone();
            nb.play_unchecked(mv);
            // Start the child's transposition line moving now. A probe is a guaranteed cache miss on a table larger than L2, and the accumulator update below needs none of it -- ~230 cycles of cover for the latency.
            self.tt.prefetch(nb.hash());
            let nacc = acc.map(|a| {
                let victim = if is_capture {
                    let p = Piece::index((tag::victim(tg) - 1) as usize);
                    let sq = if Some(mv.to) == ep_sq {
                        Square::new(mv.to.file(), mv.from.rank())
                    } else {
                        mv.to
                    };
                    Some((p, sq))
                } else {
                    None
                };
                nnue::acc_update_known(self.net.unwrap(), a, board, mv, Piece::index(tag::piece(tg)), victim)
            });
            let na = nacc.as_ref();
            self.path.push(hash);
            self.cont_stack.push(cur_ci);

            let sc = if i == 0 {
                // the first move of a PV node starts a PV node; elsewhere the expectation flips
                let child_cut = if is_pv { false } else { !cut_node };
                -self.negamax(&nb, new_depth, ply + 1, -beta, -alpha, pack(mv), na, child_cut)
            } else {
                // LMR on late quiets
                let mut r = 0i32;
                let lmr_eligible = is_quiet || (CAPT_LMR && is_capture);
                if depth >= 3 && i >= 3 && lmr_eligible && !in_check {
                    r = self.lmr[depth.min(63) as usize][i.min(63)] as i32;
                    // A late capture still changes material, so it deserves a shallower cut
                    // than a late quiet -- one ply back, never below zero.
                    if CAPT_LMR && !is_quiet {
                        r = (r - 1).max(0);
                    }
                    if LMR_TWEAKS {
                        // A PV node is worth searching more carefully; a line that is not
                        // improving is worth searching less. Both quantities are already
                        // computed for LMP, so this costs nothing.
                        r -= is_pv as i32;
                        r += !improving as i32;
                        r = r.max(0);
                    }
                    // Reduce a move the history likes by less, one the history dislikes by
                    // more. The reduction table sees only depth and move index -- it cannot
                    // tell a quiet the search keeps rewarding from one it keeps refuting, and
                    // that information is already in the tables the ordering just used.
                    if CUT_NODE && cut_node {
                        r += CUT_LMR_BONUS;
                    }
                    #[cfg(feature = "instrument")]
                    instrument::record(
                        self.history[stm as usize][mv.from as usize][mv.to as usize] as i32
                            + self.cont_score(&preds, cur_ci),
                    );
                    if HIST_LMR {
                        // F3: when the ordering score already IS history + continuation, reuse
                        // it rather than reloading both tables. Killers and countermoves are
                        // quiets whose score is 80_000 / 78_000, unrelated to their history, so
                        // they must still be computed -- the tag distinguishes them.
                        let h = if HIST_LMR_REUSE && tag::is_hist_score(tg) {
                            sc
                        } else {
                            self.history[stm as usize][mv.from as usize][mv.to as usize] as i32
                                + self.cont_score(&preds, cur_ci)
                        };
                        r -= (h / tune::hist_lmr_div()).clamp(-2, 2);
                        r = r.max(0);
                    }
                }
                // A reduced null-window search is by construction an attempt to refute the
                // move: it is a cut node.
                let mut s = -self
                    .negamax(&nb, new_depth - r, ply + 1, -alpha - 1, -alpha, pack(mv), na, true);
                if s > alpha && r > 0 {
                    s = -self.negamax(
                        &nb, new_depth, ply + 1, -alpha - 1, -alpha, pack(mv), na, !cut_node,
                    );
                }
                if s > alpha && s < beta {
                    s = -self.negamax(&nb, new_depth, ply + 1, -beta, -alpha, pack(mv), na, false);
                }
                s
            };
            self.cont_stack.pop();
            self.path.pop();
            if self.stopped {
                self.move_buf[ply as usize] = moves;
                self.quiet_buf[ply as usize] = quiets_tried;
                self.capt_buf[ply as usize] = caps_tried;
                return best.max(sc);
            }

            if sc > best {
                best = sc;
                best_mv = pack(mv);
                best_is_capture = is_capture;
                if ply == 0 {
                    self.root_best = best_mv;
                }
                if sc > alpha {
                    alpha = sc;
                    bound = BOUND_EXACT;
                    if sc >= beta {
                        bound = BOUND_LOWER;
                        #[cfg(feature = "instrument")]
                        instrument::record_cutoff(i, ord_s, tg);
                        if is_quiet {
                            // killers + history (with gravity), penalize earlier quiets
                            let k = &mut self.killers[ply as usize];
                            let packed = pack(mv);
                            if k[0] != packed {
                                k[1] = k[0];
                                k[0] = packed;
                            }
                            if prev != 0 {
                                self.counter[stm as usize][(prev & 63) as usize]
                                    [((prev >> 6) & 63) as usize] = pack(mv);
                            }
                            let bonus = hist_bonus(depth);
                            let h = &mut self.history[stm as usize][mv.from as usize]
                                [mv.to as usize];
                            let v = *h as i32;
                            *h = (v + bonus - v * bonus / 16_384) as i16;
                            self.cont_bonus(&preds, cur_ci, bonus);
                            for &(q, qci) in &quiets_tried {
                                let h = &mut self.history[stm as usize][q.from as usize]
                                    [q.to as usize];
                                let v = *h as i32;
                                *h = (v - bonus - v * bonus / 16_384) as i16;
                                // F1: qci was computed when the move was tried, not re-derived
                                self.cont_bonus(&preds, qci as usize, -bonus);
                            }
                        }
                        // Capture history, same gravity update as the quiet tables. Rewarded
                        // only when a CAPTURE caused the cutoff; the captures that were tried
                        // first and did not cut are penalised either way, because "searched
                        // ahead of the move that worked" is exactly the ordering mistake this
                        // table exists to correct.
                        if CAPT_HIST {
                            let bonus = hist_bonus(depth);
                            if is_capture {
                                if let Some(v) = capture_victim_with(board, mv, enemy, ep_sq) {
                                    let a = Piece::index(tag::piece(tg));
                                    let e = &mut self.capt[capt_index(stm, a, mv.to, v)];
                                    let cv = *e as i32;
                                    *e = (cv + bonus - cv * bonus / 16_384) as i16;
                                }
                            }
                            for &(c, ctg) in &caps_tried {
                                if let Some(v) = capture_victim_with(board, c, enemy, ep_sq) {
                                    let a = Piece::index(tag::piece(ctg));
                                    let e = &mut self.capt[capt_index(stm, a, c.to, v)];
                                    let cv = *e as i32;
                                    *e = (cv - bonus - cv * bonus / 16_384) as i16;
                                }
                            }
                        }
                        break;
                    }
                }
            }
            if is_quiet {
                quiets_tried.push((mv, cur_ci as u16));
            } else if CAPT_HIST && is_capture {
                caps_tried.push((mv, tg));
            }
        }

        // never store a verification search: its score is for the position MINUS one move
        if excl == 0 {
            self.tt
                .store(hash, best_mv, tt_score_to(best, ply), depth, bound);
            // same exclusion, same reason: a verification score describes a different position,
            // and `in_check` is already known false here (checks went to qsearch at the top).
            if !in_check {
                self.update_corr(corr_ci, static_eval, best, bound, depth, best_is_capture);
            }
        }
        // Give the buffers back for reuse, including across successive go commands.
        self.move_buf[ply as usize] = moves;
        self.quiet_buf[ply as usize] = quiets_tried;
        self.capt_buf[ply as usize] = caps_tried;
        best
    }

    /// Mean |delta| of this side's recent root scores, in pawns. `None` until there are
    /// enough moves to measure -- we never guess from one or two points.
    fn recent_volatility(&self) -> Option<f64> {
        let h = &self.eval_hist_game;
        if h.len() < 3 {
            return None;
        }
        let n = h.len().min(VOL_W + 1);
        let w = &h[h.len() - n..];
        let mut acc = 0.0;
        for i in 1..w.len() {
            // clamp per-step: a single mate-score flip must not dominate the mean
            acc += ((w[i] - w[i - 1]).abs() as f64 / 100.0).min(5.0);
        }
        Some(acc / (w.len() - 1) as f64)
    }

    // ---------------- iterative deepening driver ----------------
    pub fn think(&mut self, board: &Board, limits: &Limits) -> Option<Move> {
        self.nodes = 0;
        self.stopped = false;
        // The caller resets the shared stop flag BEFORE launching workers. Clearing it here
        // would lose a stop/quit delivered between thread creation and search startup.
        self.start = Instant::now();
        self.path.clear();
        self.cont_stack.clear();
        self.killers = [[0; 2]; MAX_PLY];
        self.root_best = 0;
        self.net = nnue::net();
        // Rebuilt per search: under `tune` the parameters change between games.
        self.lmr = build_lmr_table();
        // only the main thread of an SMP group bumps age -- once per `go`, not once per thread
        if self.is_main {
            self.tt.new_search();
        }

        // time budget
        self.max_nodes = limits.nodes.unwrap_or(u64::MAX);
        let (soft, hard) = if let Some(mt) = limits.movetime {
            (mt, mt)
        } else {
            let (t, inc) = match board.side_to_move() {
                Color::White => (limits.wtime, limits.winc.unwrap_or(0)),
                Color::Black => (limits.btime, limits.binc.unwrap_or(0)),
            };
            match t {
                Some(t) => {
                    allocate_time(t.saturating_sub(limits.move_overhead), inc, limits.movestogo)
                }
                None => (u128::MAX, u128::MAX),
            }
        };
        // Scale the soft limit by how much this side's eval has been moving lately: think
        // longer when the game is sharp, shorter when it is quiet. The hard limit is left
        // alone -- it is the safety net against flagging and must not stretch, so the scaled
        // soft limit is clamped to it.
        let soft = if VOL_TM && !limits.infinite && limits.movetime.is_none() {
            match self.recent_volatility() {
                Some(v) => {
                    let x = ((v - VOL_LO) / (VOL_HI - VOL_LO)).clamp(0.0, 1.0);
                    let f = VOL_MIN + x * (VOL_MAX - VOL_MIN);
                    (((soft as f64) * f) as u128).min(hard)
                }
                None => soft,
            }
        } else {
            soft
        };
        self.soft_ms = if limits.infinite { u128::MAX } else { soft };
        self.hard_ms = if limits.infinite { u128::MAX } else { hard };

        let max_depth = limits.depth.unwrap_or(MAX_PLY as i32 - 1).min(MAX_PLY as i32 - 1);

        // root accumulator: Some iff a net is loaded (the invariant eval_node relies on)
        let root_acc = nnue::net().map(|n| nnue::acc_from(n, board));

        let mut best: Option<Move> = None;
        let mut prev_score = 0i32;
        let mut previous_best = None;
        let mut stable_iterations = 0u32;
        let adaptive = limits.adaptive_time && !limits.infinite
            && limits.movetime.is_none() && limits.nodes.is_none() && limits.depth.is_none()
            && match board.side_to_move() {
                Color::White => limits.wtime.is_some(),
                Color::Black => limits.btime.is_some(),
            };

        // Lazy SMP diversity: each helper starts its iterative deepening at a different depth,
        // so the threads explore different trees instead of running identical work. Thread 0
        // (main, and every single-threaded caller) starts at 1, so Threads=1 is byte-identical.
        let depth0 = 1 + (self.thread_id % 3) as i32;
        for depth in depth0..=max_depth {
            self.seldepth = 0;
            self.root_depth = depth;
            // aspiration windows after depth 5
            let mut delta = 30;
            let (mut a, mut b) = if depth >= 5 {
                (prev_score - delta, prev_score + delta)
            } else {
                (-INF, INF)
            };
            let score = loop {
                let s = self.negamax(board, depth, 0, a, b, 0, root_acc.as_ref(), false);
                if self.stopped {
                    break s;
                }
                if s <= a {
                    a -= delta;
                    delta *= 2;
                } else if s >= b {
                    b += delta;
                    delta *= 2;
                } else {
                    break s;
                }
            };

            // 2026-08-18 FIX: take the root move from what the root search actually returned,
            // not by re-probing the TT. The replacement policy triggers on `existing_key != key`
            // alone, so the root entry can be evicted mid-search by any colliding position, and
            // the re-probe would then hand back a different (or no) move than the search chose.
            // Still validated against generated moves, so an illegal move remains impossible.
            if !self.stopped || best.is_none() {
                if let Some(mv) = packed_to_move(board, self.root_best) {
                    best = Some(mv);
                }
            }
            if self.stopped {
                break;
            }
            if adaptive && depth >= 5 {
                let changed = previous_best.is_some() && previous_best != best;
                stable_iterations = if changed { 0 } else { stable_iterations + 1 };
                self.soft_ms = adaptive_soft_time(soft, hard, stable_iterations, changed,
                                                  prev_score.saturating_sub(score));
            }
            previous_best = best;
            prev_score = score;
            self.last_score = score;

            if !self.silent {
                let ms = self.start.elapsed().as_millis().max(1);
                let nps = (self.nodes as u128 * 1000 / ms) as u64;
                let score_str = if score.abs() > MATE_BOUND {
                    let mate_in = (MATE - score.abs() + 1) / 2;
                    format!("mate {}", if score > 0 { mate_in } else { -mate_in })
                } else {
                    format!("cp {}", score)
                };
                let pv = self.pv_string(board, depth);
                println!(
                    "info depth {} seldepth {} score {} nodes {} nps {} time {} pv{}",
                    depth, self.seldepth, score_str, self.nodes, nps, ms, pv
                );
            }

            if self.start.elapsed().as_millis() > self.soft_ms {
                break;
            }
            if score.abs() > MATE_BOUND && depth > 2 * (MATE - score.abs()) {
                break; // mate found with margin; stop burning time
            }
        }
        best
    }

    fn tt_move(&self, board: &Board) -> Option<Move> {
        let e = self.tt.probe(board.hash())?;
        packed_to_move(board, e.mv)
    }

    fn pv_string(&self, board: &Board, max_len: i32) -> String {
        let mut s = String::new();
        let mut b = board.clone();
        let mut seen: Vec<u64> = Vec::new();
        for _ in 0..max_len {
            // stop at any repetition w.r.t. PV-internal or game history, or at the
            // fifty-move mark (display only; avoids emitting PV moves past a drawn
            // position, which match runners flag)
            if seen.contains(&b.hash()) || self.game_hist.contains(&b.hash())
                || b.halfmove_clock() >= 100
            {
                break;
            }
            seen.push(b.hash());
            match self.tt_move(&b) {
                Some(mv) => {
                    s.push(' ');
                    s.push_str(&cozy_chess::util::display_uci_move(&b, mv).to_string());
                    b.play_unchecked(mv);
                }
                None => break,
            }
        }
        s
    }
}

/// Resolve a packed move against the legal moves of `board`. Returns None if it matches none,
/// which is the invariant that keeps a TT collision or a stale root entry from ever producing
/// an illegal bestmove.
fn packed_to_move(board: &Board, packed: u16) -> Option<Move> {
    if packed == 0 {
        return None;
    }
    let mut found = None;
    board.generate_moves(|pm| {
        for mv in pm {
            if pack(mv) == packed {
                found = Some(mv);
                return true;
            }
        }
        false
    });
    found
}

/// Index into the capture-history table.
#[inline]
fn capt_index(stm: Color, piece: Piece, to: Square, victim: Piece) -> usize {
    (((stm as usize) * 384) + (piece as usize) * 64 + to as usize) * 6 + victim as usize
}

/// History bonus for a cutoff at `depth`. See HIST_BONUS.
#[inline]
fn hist_bonus(depth: i32) -> i32 {
    if HIST_BONUS {
        (depth * depth * tune::hist_bonus_mul()).min(tune::hist_bonus_max())
    } else {
        (depth * depth).min(400)
    }
}

/// How many continuation-history blocks are live. Two unless CONT_EXTRA is on.
#[inline]
const fn cont_blocks() -> usize {
    if CONT_EXTRA { CONT_BLOCKS } else { 2 }
}

fn adaptive_soft_time(base: u128, hard: u128, stable: u32, changed: bool, drop_cp: i32) -> u128 {
    // Conservative first experiment in Patzer's score units. Recompute from the
    // original budget, never compound multipliers across depths or extend hard time.
    let stability = if changed { 120 } else if stable >= 3 { 75 } else { 100 };
    let falling = drop_cp.clamp(0, 100) as u128 * 30 / 100;
    (base.saturating_mul(stability + falling) / 100).max(1).min(hard)
}

#[cfg(test)]
mod adaptive_time_tests {
    use super::*;

    #[test]
    fn stable_moves_save_time_and_deterioration_spends_it() {
        assert!(adaptive_soft_time(1000, 3000, 4, false, 0) < 1000);
        assert!(adaptive_soft_time(1000, 3000, 0, true, 80) > 1000);
        assert_eq!(adaptive_soft_time(1000, 3000, 0, false, -500), 1000);
    }

    #[test]
    fn a_full_block_retains_its_hard_reserve() {
        for stressed in [false, true] {
            let mut remaining = 900_000u128;
            for moves in (1..=40).rev() {
                let (base, hard) = allocate_time(remaining, 0, Some(moves));
                let soft = adaptive_soft_time(base, hard, if stressed { 0 } else { 4 },
                                               stressed, if stressed { 100 } else { 0 });
                assert!(soft > 0 && soft <= hard && hard < remaining);
                let spend = (soft * 14 / 10).min(hard);
                remaining -= spend;
                assert!(remaining > 0);
            }
        }
    }
}

/// (soft, hard) time budget in ms for one move.
///
/// `soft` is checked between iterative-deepening iterations; `hard` is the safety net inside
/// the search. The bench signature CANNOT gate this -- bench runs at fixed depth, so time
/// management never engages and the signature is identical however this behaves.
/// `search::tm_tests` is the gate instead.
///
/// 2026-09-09: MOVES-TO-GO, for CCRL 40/15. `t / 30` is an increment-control rule: it spends a
/// thirtieth of what is LEFT, decaying geometrically -- move 1 gets thirty seconds, move 40 gets
/// eight. A moves-to-go control states exactly how many moves the budget must cover, so spend
/// the fair share of it instead.
///
/// SIZE OF THE EFFECT, STATED HONESTLY. The engine finishes the iteration in progress and so
/// overshoots `soft` -- measured ~1.25x here (25.4 s against a 20.2 s target). That overshoot
/// PARTLY COMPENSATES for the old rule's underspending, so the gain is smaller than an
/// exact-spend model suggests. Budget used across a 40-move block, old rule -> this one:
///
///     overshoot 1.00x   74.2% -> 99.6%    (+27.6 Elo at 65/doubling)
///     overshoot 1.25x   81.8% -> 100.0%   (+18.9)   <- the measured operating point
///     overshoot 1.70x   90.3% -> 100.0%   (+9.5)
///
/// so ~+19 Elo is the number to expect, not the +28 the naive model gives.
fn allocate_time(t: u128, inc: u128, movestogo: Option<u32>) -> (u128, u128) {
    // Reserve. A FLAT 100 ms is not enough across a whole block: the engine finishes the
    // iteration in progress, so it overshoots `soft` by ~25% (measured: 25.4 s against a
    // 20.2 s target at 40/15), and while the fair-share recomputation absorbs that without
    // flagging, it lands the block with only the reserve left. Scaling by the moves still to
    // play buys a cushion where it is needed -- early, when many round-trips remain -- and
    // gives it back on the last move, where only one is left. Capped at 5% of the clock so it
    // can never dominate the budget.
    // 150 ms floor plus 20 ms per remaining move, capped at 5% of the clock. The floor is what
    // matters: a purely proportional reserve shrinks to nothing on the LAST move of a block,
    // which is exactly where a flag costs the game.
    let reserve = |t: u128, mtg: u128| 150 + (20 * mtg).min(t / 20);
    let (soft, hard) = match movestogo {
        Some(mtg) => {
            let mtg = mtg.max(1) as u128;
            let avail = t.saturating_sub(reserve(t, mtg)).max(1);
            let fair = avail / mtg;
            // Spend a shade under the fair share so the surplus rolls forward: the block then
            // finishes near 100% of the budget instead of flagging on the last move.
            let soft = fair * 9 / 10 + inc * 3 / 4;
            // A burst is allowed for a hard move, never past what the rest of the block needs.
            let hard = (fair * 3).min(avail).max(soft);
            (soft, hard)
        }
        None => {
            let soft = t / 30 + inc * 3 / 4;
            let hard = (t / 6).max(soft).min(t.saturating_sub(50)).max(1);
            (soft, hard)
        }
    };
    let hard = hard.min(t.saturating_sub(50)).max(1);
    (soft.min(hard), hard)
}

/// Per-move tag carried beside the ordering score, so the search loop never re-derives what the
/// scoring closure already knew. FREE: `(i32, Move, u8)` is 8 bytes -- exactly what `(i32, Move)`
/// already occupied, since the old tuple wasted a byte to padding.
///
///   bits 0..2  the moving piece, `Piece as u8`
///   bit 3      capture (including en passant)
///   bit 4      `see(board, mv) >= 0`; meaningful only for captures
mod tag {
    pub const CAPTURE: u8 = 1 << 3;
    pub const SEE_WIN: u8 = 1 << 4;
    /// The ordering score for this move IS `history + cont_score`, so HIST_LMR can reuse it.
    /// NOT merely "is a quiet": killers score 80_000 and countermoves 78_000, and a quiet that
    /// took one of those arms has a score unrelated to its history. Reusing `s` for those would
    /// drive the LMR adjustment straight to its clamp.
    pub const IS_HIST_SCORE: u8 = 1 << 5;
    #[inline]
    pub fn piece(t: u8) -> usize { (t & 7) as usize }
    #[inline]
    pub fn is_capture(t: u8) -> bool { t & CAPTURE != 0 }
    #[inline]
    pub fn see_winning(t: u8) -> bool { t & SEE_WIN != 0 }
    #[inline]
    pub fn is_hist_score(t: u8) -> bool { t & IS_HIST_SCORE != 0 && t & CAPTURE == 0 }
    /// victim + 1, valid only when `is_capture(t)`; 0 means no victim. Shares bits 5..7 with
    /// IS_HIST_SCORE, which is only ever set on quiets.
    #[inline]
    pub fn victim(t: u8) -> u8 { (t >> 5) & 7 }
}

fn has_non_pawn(board: &Board, c: Color) -> bool {
    let mine = board.colors(c);
    !((board.pieces(Piece::Knight)
        | board.pieces(Piece::Bishop)
        | board.pieces(Piece::Rook)
        | board.pieces(Piece::Queen))
        & mine)
        .is_empty()
}

fn tt_score_to(s: i32, ply: i32) -> i32 {
    if s > MATE_BOUND {
        s + ply
    } else if s < -MATE_BOUND {
        s - ply
    } else {
        s
    }
}

fn tt_score_from(s: i32, ply: i32) -> i32 {
    if s > MATE_BOUND {
        s - ply
    } else if s < -MATE_BOUND {
        s + ply
    } else {
        s
    }
}

/// Gate for the 2026-08-18 en-passant classification fix.
///
/// The bench signature cannot certify this: en passant is rare enough that a bench position set
/// may contain none at all, and a signature only says "the tree changed", never "it changed for
/// the right reason". These assert the classification directly, including the castling case that
/// a naive "destination not occupied by the enemy" rule would get wrong in the other direction.
#[cfg(test)]
mod ep_tests {
    use super::*;

    #[test]
    fn search_start_preserves_a_pending_stop() {
        let mut s = Searcher::new(1);
        s.silent = true;
        s.stop.store(true, Ordering::Relaxed);
        assert!(s.think(&Board::startpos(), &Limits::default()).is_none());
        assert_eq!(s.nodes, 0);
        assert!(s.stop.load(Ordering::Relaxed));
    }

    #[test]
    fn interrupted_search_preserves_scratch_capacity_for_next_move() {
        let mut s = Searcher::new(1);
        s.silent = true;
        for _ in 0..2 {
            s.think(&Board::startpos(), &Limits { nodes: Some(2048), ..Default::default() });
            assert!(s.stopped);
            assert!(s.move_buf.iter().all(|v| v.capacity() >= 64));
            assert!(s.qmove_buf.iter().all(|v| v.capacity() >= 32));
            assert!(s.quiet_buf.iter().all(|v| v.capacity() >= 64));
            assert!(s.capt_buf.iter().all(|v| v.capacity() >= 32));
        }
    }

    fn find(board: &Board, uci: &str) -> Move {
        let mut found = None;
        board.generate_moves(|pm| {
            for mv in pm {
                if cozy_chess::util::display_uci_move(board, mv).to_string() == uci {
                    found = Some(mv);
                    return true;
                }
            }
            false
        });
        found.unwrap_or_else(|| panic!("move {uci} not legal here"))
    }

    /// The old predicate, kept verbatim so the test states exactly what regressed.
    fn old_is_capture(board: &Board, mv: Move) -> bool {
        board.color_on(mv.to) == Some(opp(board.side_to_move()))
    }

    #[test]
    fn en_passant_is_a_capture() {
        // white pawn e5, black pawn d5, ep target d6: exd6 e.p. takes the d5 pawn
        let b = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", false).unwrap();
        let ep = find(&b, "e5d6");
        assert_eq!(capture_victim(&b, ep), Some(Piece::Pawn));
        // the bug: the destination square is empty, so the old test called it quiet
        assert!(!old_is_capture(&b, ep), "test is not exercising the bug");
    }

    #[test]
    fn quiet_pawn_push_is_not_a_capture() {
        let b = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", false).unwrap();
        assert_eq!(capture_victim(&b, find(&b, "e5e6")), None);
    }

    #[test]
    fn castling_is_not_a_capture() {
        // cozy encodes castling as king-takes-own-rook, so `to` IS occupied -- by our own rook.
        let b = Board::from_fen("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", false).unwrap();
        assert_eq!(capture_victim(&b, find(&b, "e1g1")), None);
        assert_eq!(capture_victim(&b, find(&b, "e1c1")), None);
    }

    #[test]
    fn ordinary_capture_still_reads_its_victim() {
        let b = Board::from_fen("4k3/8/8/3r4/4B3/8/8/4K3 w - - 0 1", false).unwrap();
        assert_eq!(capture_victim(&b, find(&b, "e4d5")), Some(Piece::Rook));
    }

    /// qsearch generates a move only when `capture_victim` (or a queen promo, or being in check)
    /// says so, which is the path that used to drop en passant entirely.
    #[test]
    fn qsearch_now_sees_the_en_passant_capture() {
        let b = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", false).unwrap();
        let ep = find(&b, "e5d6");
        let in_check = !b.checkers().is_empty();
        let generated = in_check || capture_victim(&b, ep).is_some()
            || ep.promotion == Some(Piece::Queen);
        assert!(generated, "qsearch would still skip the en passant capture");
    }
}

#[cfg(test)]
mod vol_tm_tests {
    use super::*;

    fn searcher_with(hist: Vec<i32>) -> Searcher {
        let mut s = Searcher::new(1);
        s.eval_hist_game = hist;
        s
    }

    /// The bench signature runs at FIXED DEPTH, so it never engages time management and is
    /// blind to this feature. These tests are the gate instead.

    #[test]
    fn too_short_a_history_declines_to_guess() {
        assert_eq!(searcher_with(vec![]).recent_volatility(), None);
        assert_eq!(searcher_with(vec![10]).recent_volatility(), None);
        assert_eq!(searcher_with(vec![10, 20]).recent_volatility(), None);
        // three points = two deltas is the first measurable case
        assert!(searcher_with(vec![10, 20, 30]).recent_volatility().is_some());
    }

    #[test]
    fn volatility_is_mean_abs_delta_in_pawns() {
        // deltas of 50cp and 150cp -> mean 100cp -> 1.0 pawns
        let v = searcher_with(vec![0, 50, 200]).recent_volatility().unwrap();
        assert!((v - 1.0).abs() < 1e-9, "got {v}");
        // a flat eval is zero volatility
        let v = searcher_with(vec![30, 30, 30, 30]).recent_volatility().unwrap();
        assert!(v.abs() < 1e-9, "got {v}");
    }

    #[test]
    fn a_single_mate_flip_cannot_dominate() {
        // one 320-pawn jump clamps to 5.0 pawns, so the mean stays bounded
        let v = searcher_with(vec![0, 0, 32000]).recent_volatility().unwrap();
        assert!(v <= 2.5 + 1e-9, "mate flip leaked through: {v}");
    }

    #[test]
    fn window_is_bounded_by_vol_w() {
        // ancient history must not matter: a long calm tail after old chaos reads calm
        let mut h = vec![0, 30000];
        h.extend(std::iter::repeat(500).take(VOL_W + 4));
        let v = searcher_with(h).recent_volatility().unwrap();
        assert!(v.abs() < 1e-9, "window not truncated to VOL_W: {v}");
    }

    #[test]
    fn scaling_factor_spans_min_to_max_and_clamps() {
        let f = |v: f64| {
            let x = ((v - VOL_LO) / (VOL_HI - VOL_LO)).clamp(0.0, 1.0);
            VOL_MIN + x * (VOL_MAX - VOL_MIN)
        };
        assert!((f(0.0) - VOL_MIN).abs() < 1e-9); // calmer than LO -> clamped
        assert!((f(VOL_LO) - VOL_MIN).abs() < 1e-9);
        assert!((f(VOL_HI) - VOL_MAX).abs() < 1e-9);
        assert!((f(99.0) - VOL_MAX).abs() < 1e-9); // wilder than HI -> clamped
        assert!(f(0.35) > VOL_MIN && f(0.35) < VOL_MAX); // monotone in between
    }
}

/// Gate for the 2026-09-09 qsearch destination mask.
///
/// The mask's whole licence to ship without an SPRT is that it changes NO move list -- it only
/// declines to construct moves the filter was about to discard. The bench signature says the
/// tree is unchanged on six positions; this says the move list is unchanged on thousands,
/// including the two cases a destination mask is most likely to get wrong (an en-passant
/// capture, whose destination square is empty, and a non-capturing promotion, whose destination
/// is empty too).
#[cfg(test)]
mod qmask_tests {
    use super::*;

    /// Moves qsearch would search at `board`, built WITHOUT the mask -- the pre-2026-09-09 path.
    fn unmasked(board: &Board) -> Vec<Move> {
        let in_check = !board.checkers().is_empty();
        let mut out = Vec::new();
        board.generate_moves(|pm| {
            for mv in pm {
                let victim = capture_victim(board, mv);
                let is_qpromo = mv.promotion == Some(Piece::Queen);
                if in_check || victim.is_some() || is_qpromo {
                    out.push(mv);
                }
            }
            false
        });
        out
    }

    /// The same list, built the way qsearch builds it now.
    fn masked(board: &Board) -> Vec<Move> {
        let stm = board.side_to_move();
        let in_check = !board.checkers().is_empty();
        let qtargets = if in_check {
            BitBoard::FULL
        } else {
            let mut t = board.colors(opp(stm));
            if let Some(f) = board.en_passant() {
                t |= Square::new(f, Rank::Sixth.relative_to(stm)).bitboard();
            }
            t
        };
        let promo_rank = Rank::Eighth.relative_to(stm).bitboard();
        let mut out = Vec::new();
        board.generate_moves(|mut pm| {
            if !in_check {
                let mut t = qtargets;
                if pm.piece == Piece::Pawn {
                    t |= promo_rank;
                }
                pm.to &= t;
                if pm.to.is_empty() {
                    return false;
                }
            }
            for mv in pm {
                let victim = capture_victim(board, mv);
                let is_qpromo = mv.promotion == Some(Piece::Queen);
                if in_check || victim.is_some() || is_qpromo {
                    out.push(mv);
                }
            }
            false
        });
        out
    }

    fn agree(board: &Board, what: &str) {
        assert_eq!(masked(board), unmasked(board), "qsearch move list changed: {what}");
    }

    #[test]
    fn mask_preserves_the_move_list_on_random_games() {
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut positions = 0usize;
        let mut saw_ep = 0usize;
        let mut saw_promo = 0usize;
        for _ in 0..300 {
            let mut board = Board::startpos();
            for _ in 0..120 {
                if board.status() != cozy_chess::GameStatus::Ongoing {
                    break;
                }
                agree(&board, "random game");
                positions += 1;
                if board.en_passant().is_some() {
                    saw_ep += 1;
                }
                let mut legal = Vec::new();
                board.generate_moves(|pm| {
                    for mv in pm {
                        if mv.promotion.is_some() {
                            saw_promo += 1;
                        }
                        legal.push(mv);
                    }
                    false
                });
                let mv = legal[(rnd() as usize) % legal.len()];
                board.play_unchecked(mv);
            }
        }
        assert!(positions > 10_000, "only {positions} positions walked");
        assert!(saw_ep > 50, "only {saw_ep} en-passant positions -- test is not covering ep");
        assert!(saw_promo > 50, "only {saw_promo} promotions -- test is not covering promos");
    }

    /// The two shapes a destination mask gets wrong, stated as positions rather than left to
    /// chance in the random walk.
    #[test]
    fn mask_keeps_en_passant_and_quiet_promotions() {
        // en passant: destination d6 is EMPTY, so it is not in the enemy occupancy
        let b = Board::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", false).unwrap();
        assert_eq!(masked(&b).len(), 1, "the ep capture was masked away");
        agree(&b, "en passant");
        // non-capturing queen promotion: destination is empty and not enemy-occupied.
        // Black king on g8, not e8: a d7 pawn attacks e8, and the side NOT to move may not be
        // left in check -- cozy rejects that FEN outright.
        let b = Board::from_fen("6k1/3P4/8/8/8/8/8/4K3 w - - 0 1", false).unwrap();
        assert!(!masked(&b).is_empty(), "the quiet queen promotion was masked away");
        agree(&b, "quiet promotion");
        // in check: every evasion must survive, quiet ones included
        let b = Board::from_fen("4k3/8/8/8/7b/8/6P1/4K2R w K - 0 1", false).unwrap();
        agree(&b, "in check");
    }
}

/// Gate for moves-to-go time management (2026-09-09). The bench signature is blind to this --
/// fixed-depth search never consults the clock -- so these assertions are the only thing
/// standing between a mis-specified budget and a forfeited game.
#[cfg(test)]
mod tm_tests {
    use super::*;

    /// Walk a whole CCRL 40/15 block and report (fraction of budget used, worst overspend).
    fn walk_block(movestogo: bool) -> (f64, i128) {
        let budget: u128 = 900_000;
        let mut t = budget;
        let mut used: u128 = 0;
        // MIN, not 0: seeded at 0 the `max` below can only grow it, so `worst < 0` would be
        // unfalsifiable and the assertion would fire on a correct allocator (it did).
        let mut worst: i128 = i128::MIN;
        for mv in 0..40u32 {
            let mtg = if movestogo { Some(40 - mv) } else { None };
            let (soft, hard) = allocate_time(t, 0, mtg);
            assert!(soft <= hard, "soft {soft} exceeded hard {hard} at move {mv}");
            // The engine finishes the iteration in progress, so it overshoots `soft`. 1.4x is
            // above the ~1.25x measured at 40/15, deliberately: the budget has to survive the
            // bad case, not the typical one.
            let spend = (soft * 14 / 10).min(hard);
            worst = worst.max(spend as i128 - t as i128);
            used += spend;
            t = t.saturating_sub(spend);
        }
        (used as f64 / budget as f64, worst)
    }

    #[test]
    fn moves_to_go_spends_the_block() {
        let (frac, worst) = walk_block(true);
        assert!(worst < 0, "a move was allocated more time than remained on the clock");
        // Measured 99.6% with the 9/10 fair-share rule: the shortfall each move rolls into
        // the remaining ones, so the block finishes nearly exact without ever overcommitting.
        assert!(frac > 0.90, "only {:.1}% of the 40/15 budget used", 100.0 * frac);
        assert!(frac < 1.0, "allocated {:.1}% of the budget -- that flags", 100.0 * frac);
    }

    /// The regression this replaced: without movestogo the same block leaves a quarter of the
    /// clock unused. Kept as a test so the improvement is a measured fact, not a claim.
    #[test]
    fn the_increment_rule_underspends_a_moves_to_go_block() {
        let (frac, _) = walk_block(false);
        // Both arms carry the SAME overshoot model. That matters: overshooting partly
        // compensates for the old rule's underspending, so applying it to only one arm would
        // overstate the gain -- which an earlier version of this test did.
        assert!(frac < 0.90, "expected the old rule to underspend, got {:.1}%", 100.0 * frac);
        let (with_mtg, _) = walk_block(true);
        assert!(with_mtg > frac + 0.10, "movestogo must recover a large part of the budget");
    }

    /// Sudden-death and increment controls must be untouched by the change.
    #[test]
    fn increment_controls_are_unchanged() {
        for &(t, inc) in &[(300_000u128, 3_000u128), (10_000, 100), (40, 0)] {
            let (soft, hard) = allocate_time(t, inc, None);
            assert_eq!(soft, (t / 30 + inc * 3 / 4).min(hard));
            assert!(soft <= hard && hard >= 1);
        }
    }

    /// The last move of a block may use nearly everything, but never all of it.
    #[test]
    fn the_final_move_of_a_block_keeps_a_reserve() {
        let (soft, hard) = allocate_time(120_000, 0, Some(1));
        assert!(hard < 120_000, "no reserve left against the clock");
        assert!(soft > 90_000, "far too timid with one move to make: {soft} ms");
    }

    /// A nearly-flagged clock must still return something playable rather than 0.
    #[test]
    fn a_desperate_clock_still_yields_a_positive_budget() {
        for &t in &[1u128, 40, 120, 1000] {
            for mtg in [None, Some(1), Some(40)] {
                let (soft, hard) = allocate_time(t, 0, mtg);
                assert!(hard >= 1 && soft <= hard, "t={t} mtg={mtg:?} -> ({soft},{hard})");
            }
        }
    }
}
