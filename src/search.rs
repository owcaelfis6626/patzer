//! Stage-1.5 classical search: iterative deepening, aspiration windows, PVS negamax (fail-soft),
//! transposition table, quiescence with in-check evasions + SEE pruning, null-move pruning,
//! reverse futility, LMR, ordering = TT move > SEE-winning captures (MVV-LVA) > queen promos >
//! killers > countermove > (butterfly history + continuation history) > SEE-losing captures.
//! Continuation history (Stage 4): (piece,to) of the 1-ply and 2-ply predecessors index a
//! histogram summed into the residual quiet score; same gravity update as butterfly history.
//! Single thread.

use crate::eval::{evaluate, piece_val};
use crate::nnue;
use crate::tt::{pack, BOUND_EXACT, BOUND_LOWER, BOUND_UPPER, TT};
use cozy_chess::{
    get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks, get_rook_moves,
    BitBoard, Board, Color, Move, Piece, Square,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

pub const MATE: i32 = 32_000;
pub const INF: i32 = 32_500;
const MATE_BOUND: i32 = MATE - 512;
const MAX_PLY: usize = 128;
const CONT_PT: usize = 6 * 64; // continuation-history (piece,to) index space = 384
const NULL_CONT: usize = usize::MAX; // sentinel: no continuation across a null move

fn opp(c: Color) -> Color {
    match c {
        Color::White => Color::Black,
        Color::Black => Color::White,
    }
}

// ---------------- static exchange evaluation ----------------
// Swap algorithm with x-ray updates (sliders recomputed on the reduced occupancy each iteration).
// Approximations (standard for ordering): promotions valued as the moving pawn; a king may not
// "capture into" remaining enemy attackers (loop stops there).

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
    let target = mv.to;
    let attacker = match board.piece_on(mv.from) {
        Some(p) => p,
        None => return 0,
    };
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

fn lmr_table() -> &'static [[i8; 64]; 64] {
    use std::sync::OnceLock;
    static LMR: OnceLock<[[i8; 64]; 64]> = OnceLock::new();
    LMR.get_or_init(|| {
        let mut t = [[0i8; 64]; 64];
        for (d, row) in t.iter_mut().enumerate().skip(1) {
            for (m, r) in row.iter_mut().enumerate().skip(1) {
                *r = (0.75 + (d as f64).ln() * (m as f64).ln() / 2.25) as i8;
            }
        }
        t
    })
}

#[derive(Clone, Default)]
pub struct Limits {
    pub depth: Option<i32>,
    pub nodes: Option<u64>,
    pub movetime: Option<u128>,
    pub wtime: Option<u128>,
    pub btime: Option<u128>,
    pub winc: Option<u128>,
    pub binc: Option<u128>,
    pub infinite: bool,
}

pub struct Searcher {
    pub tt: TT,
    killers: [[u16; 2]; MAX_PLY],
    history: [[[i32; 64]; 64]; 2],
    counter: [[[u16; 64]; 64]; 2], // [stm][prev.from][prev.to] -> packed countermove
    // continuation history: two offset blocks (1-ply-ago, 2-ply-ago), each indexed
    // [prev_piece*64+prev_to][cur_piece*64+cur_to]; flat to avoid a large on-stack array.
    cont: Vec<i32>,
    cont_stack: Vec<usize>, // (piece,to) index of each move on the path; NULL_CONT for null moves
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
    pub silent: bool,
    pub last_score: i32, // score of the last completed ID iteration (stm perspective)
}

impl Searcher {
    pub fn new(hash_mb: usize) -> Self {
        Searcher {
            tt: TT::new(hash_mb),
            killers: [[0; 2]; MAX_PLY],
            history: [[[0; 64]; 64]; 2],
            counter: [[[0; 64]; 64]; 2],
            cont: vec![0i32; 2 * CONT_PT * CONT_PT],
            cont_stack: Vec::with_capacity(MAX_PLY + 4),
            nodes: 0,
            seldepth: 0,
            start: Instant::now(),
            soft_ms: u128::MAX,
            hard_ms: u128::MAX,
            max_nodes: u64::MAX,
            stop: Arc::new(AtomicBool::new(false)),
            stopped: false,
            path: Vec::with_capacity(MAX_PLY + 4),
            game_hist: Vec::new(),
            silent: false,
            last_score: 0,
        }
    }

    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
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

    fn is_repetition(&self, hash: u64) -> bool {
        self.path.iter().rev().any(|&h| h == hash) || self.game_hist.iter().any(|&h| h == hash)
    }

    /// static eval at a node: NNUE (incremental accumulator) when a net is loaded, else PeSTO.
    /// Invariant: `acc` is Some iff a net is loaded (established at the root in think()).
    #[inline]
    fn eval_node(&self, board: &Board, acc: Option<&nnue::Acc>) -> i32 {
        match acc {
            Some(a) => nnue::forward(nnue::net().unwrap(), &a.w, &a.b, board.side_to_move()),
            None => evaluate(board),
        }
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

        let mut best;
        if in_check {
            best = -INF; // must search all evasions; no stand-pat while in check
        } else {
            best = self.eval_node(board, acc);
            if best >= beta {
                return best;
            }
            alpha = alpha.max(best);
        }

        let mut moves: Vec<(i32, Move)> = Vec::with_capacity(32);
        board.generate_moves(|pm| {
            for mv in pm {
                let victim = if board.color_on(mv.to) == Some(opp(stm)) {
                    board.piece_on(mv.to)
                } else {
                    None
                };
                let is_qpromo = mv.promotion == Some(Piece::Queen);
                if in_check || victim.is_some() || is_qpromo {
                    // SEE pruning: skip losing captures entirely (not while in check)
                    if !in_check && victim.is_some() && !is_qpromo && see(board, mv) < 0 {
                        continue;
                    }
                    let attacker = board.piece_on(mv.from).unwrap();
                    let mut s = 0;
                    if let Some(v) = victim {
                        s += 100_000 + 10 * piece_val(v) - piece_val(attacker);
                    }
                    if is_qpromo {
                        s += 90_000;
                    }
                    moves.push((s, mv));
                }
            }
            false
        });

        if in_check && moves.is_empty() {
            return -MATE + ply;
        }
        moves.sort_unstable_by_key(|&(s, _)| -s);

        for (_, mv) in moves {
            // delta pruning: even winning this victim can't lift alpha
            if !in_check {
                if let Some(v) = board.piece_on(mv.to) {
                    if board.color_on(mv.to) == Some(opp(stm))
                        && best + piece_val(v) + 200 < alpha
                    {
                        continue;
                    }
                }
            }
            let mut nb = board.clone();
            nb.play_unchecked(mv);
            let nacc = acc.map(|a| nnue::acc_update(nnue::net().unwrap(), a, board, mv));
            let sc = -self.qsearch(&nb, ply + 1, -beta, -alpha, nacc.as_ref());
            if self.stopped {
                return best.max(sc);
            }
            if sc > best {
                best = sc;
                if sc > alpha {
                    alpha = sc;
                    if sc >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    // continuation-history update with the same gravity as butterfly history. `bonus` is signed:
    // positive rewards, negative penalizes. Applied to whichever predecessor slots exist.
    #[inline]
    fn cont_bonus(&mut self, cont1: Option<usize>, cont2: Option<usize>, ci: usize, bonus: i32) {
        if let Some(p) = cont1 {
            if p != NULL_CONT {
                let e = &mut self.cont[p * CONT_PT + ci];
                *e += bonus - *e * bonus.abs() / 16_384;
            }
        }
        if let Some(p) = cont2 {
            if p != NULL_CONT {
                let e = &mut self.cont[CONT_PT * CONT_PT + p * CONT_PT + ci];
                *e += bonus - *e * bonus.abs() / 16_384;
            }
        }
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
    ) -> i32 {
        if self.check_stop() {
            return 0;
        }
        let hash = board.hash();
        if ply > 0 {
            if board.halfmove_clock() >= 100 || self.is_repetition(hash) {
                self.nodes += 1;
                return 0;
            }
        }

        let in_check = !board.checkers().is_empty();
        if in_check {
            depth += 1; // check extension
        }
        if depth <= 0 {
            return self.qsearch(board, ply, alpha, beta, acc);
        }
        self.nodes += 1;
        if ply as usize >= MAX_PLY - 1 {
            return self.eval_node(board, acc);
        }

        // TT probe
        let mut tt_mv: u16 = 0;
        if let Some(e) = self.tt.probe(hash) {
            tt_mv = e.mv;
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

        let stm = board.side_to_move();
        let static_eval = self.eval_node(board, acc);

        // reverse futility pruning
        if !in_check && ply > 0 && depth <= 6 && static_eval - 120 * depth >= beta {
            return static_eval;
        }

        // null-move pruning
        if !in_check
            && ply > 0
            && depth >= 3
            && static_eval >= beta
            && has_non_pawn(board, stm)
        {
            if let Some(nb) = board.null_move() {
                let r = 3 + depth / 5;
                self.path.push(hash);
                self.cont_stack.push(NULL_CONT);
                // null move: no pieces change, accumulator carries over unchanged
                let sc = -self.negamax(&nb, depth - 1 - r, ply + 1, -beta, -beta + 1, 0, acc);
                self.cont_stack.pop();
                self.path.pop();
                if self.stopped {
                    return 0;
                }
                if sc >= beta {
                    return beta;
                }
            }
        }

        // continuation-history predecessors for this node: 1-ply-ago (counter-move history) and
        // 2-ply-ago (follow-up history). cont_stack.len() == ply here (each ply pushes once).
        let cont1 = self.cont_stack.last().copied();
        let cont2 = if self.cont_stack.len() >= 2 {
            Some(self.cont_stack[self.cont_stack.len() - 2])
        } else {
            None
        };

        // generate + score. Arm A (Stage 3): when a policy net is loaded, QUIET moves are
        // ranked by the policy logit instead of killers/countermove/history — captures,
        // promotions and the TT move keep their classical slots. act512 computed once per
        // node from the same accumulator the value head uses.
        let pol = crate::policy::policy();
        let act = match (pol, acc) {
            (Some(_), Some(a)) => Some(crate::policy::activations(a, stm)),
            _ => None,
        };
        let mut moves: Vec<(i32, Move)> = Vec::with_capacity(48);
        board.generate_moves(|pm| {
            for mv in pm {
                let packed = pack(mv);
                let is_capture = board.color_on(mv.to) == Some(opp(stm));
                let s = if packed == tt_mv && tt_mv != 0 {
                    1_000_000
                } else if is_capture {
                    let v = board.piece_on(mv.to).unwrap();
                    let a = board.piece_on(mv.from).unwrap();
                    let mvvlva = 10 * piece_val(v) - piece_val(a);
                    if see(board, mv) >= 0 {
                        100_000 + mvvlva // winning/equal captures ahead of everything but TT
                    } else {
                        -20_000 + mvvlva / 10 // losing captures behind all quiets
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
                    let mover = board.piece_on(mv.from).unwrap();
                    let ci = mover as usize * 64 + mv.to as usize;
                    let mut hsc = self.history[stm as usize][mv.from as usize][mv.to as usize];
                    if let Some(p) = cont1 {
                        if p != NULL_CONT {
                            hsc += self.cont[p * CONT_PT + ci];
                        }
                    }
                    if let Some(p) = cont2 {
                        if p != NULL_CONT {
                            hsc += self.cont[CONT_PT * CONT_PT + p * CONT_PT + ci];
                        }
                    }
                    hsc
                };
                moves.push((s, mv));
            }
            false
        });

        if moves.is_empty() {
            return if in_check { -MATE + ply } else { 0 };
        }
        moves.sort_unstable_by_key(|&(s, _)| -s);

        let mut best = -INF;
        let mut best_mv: u16 = 0;
        let mut bound = BOUND_UPPER;
        let mut quiets_tried: Vec<Move> = Vec::new();

        for (i, &(_, mv)) in moves.iter().enumerate() {
            let is_capture = board.color_on(mv.to) == Some(opp(stm));
            let is_quiet = !is_capture && mv.promotion.is_none();
            // continuation-history index of the move being played (any move, capture or quiet)
            let cur_ci = board.piece_on(mv.from).unwrap() as usize * 64 + mv.to as usize;
            let mut nb = board.clone();
            nb.play_unchecked(mv);
            let nacc = acc.map(|a| nnue::acc_update(nnue::net().unwrap(), a, board, mv));
            let na = nacc.as_ref();
            self.path.push(hash);
            self.cont_stack.push(cur_ci);

            let sc = if i == 0 {
                -self.negamax(&nb, depth - 1, ply + 1, -beta, -alpha, pack(mv), na)
            } else {
                // LMR on late quiets
                let mut r = 0i32;
                if depth >= 3 && i >= 3 && is_quiet && !in_check {
                    r = lmr_table()[depth.min(63) as usize][i.min(63)] as i32;
                }
                let mut s =
                    -self.negamax(&nb, depth - 1 - r, ply + 1, -alpha - 1, -alpha, pack(mv), na);
                if s > alpha && r > 0 {
                    s = -self.negamax(&nb, depth - 1, ply + 1, -alpha - 1, -alpha, pack(mv), na);
                }
                if s > alpha && s < beta {
                    s = -self.negamax(&nb, depth - 1, ply + 1, -beta, -alpha, pack(mv), na);
                }
                s
            };
            self.cont_stack.pop();
            self.path.pop();
            if self.stopped {
                return best.max(sc);
            }

            if sc > best {
                best = sc;
                best_mv = pack(mv);
                if sc > alpha {
                    alpha = sc;
                    bound = BOUND_EXACT;
                    if sc >= beta {
                        bound = BOUND_LOWER;
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
                            let bonus = (depth * depth).min(400);
                            let h = &mut self.history[stm as usize][mv.from as usize]
                                [mv.to as usize];
                            *h += bonus - *h * bonus / 16_384;
                            self.cont_bonus(cont1, cont2, cur_ci, bonus);
                            for q in &quiets_tried {
                                let h = &mut self.history[stm as usize][q.from as usize]
                                    [q.to as usize];
                                *h -= bonus + *h * bonus / 16_384;
                                let qci =
                                    board.piece_on(q.from).unwrap() as usize * 64 + q.to as usize;
                                self.cont_bonus(cont1, cont2, qci, -bonus);
                            }
                        }
                        break;
                    }
                }
            }
            if is_quiet {
                quiets_tried.push(mv);
            }
        }

        self.tt
            .store(hash, best_mv, tt_score_to(best, ply), depth, bound);
        best
    }

    // ---------------- iterative deepening driver ----------------
    pub fn think(&mut self, board: &Board, limits: &Limits) -> Option<Move> {
        self.nodes = 0;
        self.stopped = false;
        self.stop.store(false, Ordering::Relaxed);
        self.start = Instant::now();
        self.path.clear();
        self.cont_stack.clear();
        self.killers = [[0; 2]; MAX_PLY];
        self.tt.new_search();

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
                    let soft = t / 30 + inc * 3 / 4;
                    (soft, (t / 6).max(soft).min(t.saturating_sub(50)))
                }
                None => (u128::MAX, u128::MAX),
            }
        };
        self.soft_ms = if limits.infinite { u128::MAX } else { soft };
        self.hard_ms = if limits.infinite { u128::MAX } else { hard };

        let max_depth = limits.depth.unwrap_or(MAX_PLY as i32 - 1).min(MAX_PLY as i32 - 1);

        // root accumulator: Some iff a net is loaded (the invariant eval_node relies on)
        let root_acc = nnue::net().map(|n| nnue::acc_from(n, board));

        let mut best: Option<Move> = None;
        let mut prev_score = 0i32;

        for depth in 1..=max_depth {
            self.seldepth = 0;
            // aspiration windows after depth 5
            let mut delta = 30;
            let (mut a, mut b) = if depth >= 5 {
                (prev_score - delta, prev_score + delta)
            } else {
                (-INF, INF)
            };
            let score = loop {
                let s = self.negamax(board, depth, 0, a, b, 0, root_acc.as_ref());
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

            // pick up best move from TT (root entry)
            if !self.stopped || best.is_none() {
                if let Some(mv) = self.tt_move(board) {
                    best = Some(mv);
                }
            }
            if self.stopped {
                break;
            }
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
        let mut found = None;
        board.generate_moves(|pm| {
            for mv in pm {
                if pack(mv) == e.mv {
                    found = Some(mv);
                    return true;
                }
            }
            false
        });
        found
    }

    fn pv_string(&self, board: &Board, max_len: i32) -> String {
        let mut s = String::new();
        let mut b = board.clone();
        let mut seen: Vec<u64> = Vec::new();
        for _ in 0..max_len {
            // stop at any repetition w.r.t. PV-internal or game history (display only;
            // avoids emitting PV moves past a threefold, which match runners flag)
            if seen.contains(&b.hash()) || self.game_hist.contains(&b.hash()) {
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
