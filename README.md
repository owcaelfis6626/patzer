# patzer

A UCI chess engine written from scratch in Rust — NNUE evaluation on top of a
classical alpha-beta search. Plays live on Lichess as
[@hubilipski](https://lichess.org/@/hubilipski).

## Design

**Search** (`src/search.rs`)
- Iterative deepening with aspiration windows; principal-variation search
  (fail-soft negamax)
- Transposition table (`src/tt.rs`): lock-free, four 16-byte slots per 64-byte
  cache line, evicted by depth discounted against age. Probed and stored in
  quiescence as well — but quiescence may only take a cutoff from an entry
  quiescence itself wrote (see the note in `qsearch`; the depth test there is
  load-bearing and cost a `mates` gate failure to find)
- Pruning and reductions: null move, reverse futility, forward futility,
  late-move pruning and reductions, internal iterative reduction, SEE pruning of
  losing captures and quiets, ProbCut, mate-distance pruning
- Node-type aware: an expected-fail-high node reduces harder in LMR and takes an
  extra internal-iterative-reduction ply
- Move ordering: TT move → SEE-ranked captures with capture history → queen
  promotions → killers → counter-move → butterfly plus continuation history
- Correction history: search results are folded back into the static eval,
  bucketed by pawn structure
- Time management understands both increment and **moves-to-go** controls

**Evaluation**
- NNUE (`src/nnue.rs`): 768→256×2→16→1, efficiently-updatable accumulator,
  quantized integer inference. The hidden layer uses AVX2 `vpmaddubsw`, which is
  bit-exact against the scalar reference — `patzer fwdgate` checks that on real
  positions, and `nnue::avx2_tests` checks the saturation boundary
- Falls back to a tapered hand-crafted PeSTO evaluation (`src/pesto.rs`) when no
  net is loaded

**Move generation**: [`cozy-chess`](https://crates.io/crates/cozy-chess), perft-verified.

## Gates

Correctness lives in subcommands, and each exists because something once slipped
past its absence:

```sh
patzer perft     # exact movegen truth
patzer see       # static exchange evaluation, hand-computable cases only
patzer mates     # engine's mate claims PROVEN by exhaustive walk, not trusted
patzer book      # every repertoire line legality-walked from startpos
patzer soak      # self-play stability: every move legality-checked
patzer bench     # fixed-depth node count = functional signature
patzer nnueinc   # incremental accumulator == from-scratch rebuild
patzer fwdgate   # AVX2 forward == scalar reference, on real positions
```

`bench` prints a node-count signature that must be re-recorded whenever search
behaviour changes — and, just as usefully, must NOT change when a claimed
speed-only or refactor-only change lands. Set `PATZER_BENCH_DEPTH` to measure at
other depths; a signature taken elsewhere is not comparable to a recorded one.

`--features instrument` adds `patzer histdump`, which prints the history-score
distribution at LMR sites. Every threshold denominated in history units has to be
derived from that rather than borrowed from another engine: three heuristics were
once written against the wrong scale and measured either inert or harmful.

**Development discipline**: search and evaluation changes are accepted only via
equal-time SPRT self-play with pre-registered Elo bounds — never fixed-nodes or
fixed-depth comparisons.

## Build

```sh
cargo build --release          # release
cargo build --release --features tune         # search constants as UCI options, for SPSA
cargo build --release --features instrument   # adds histdump; NOT for release
```

## Run (UCI)

```sh
./target/release/patzer
```

```
uci
setoption name EvalFile value path/to/net.nnue   # optional — PeSTO eval used if omitted
setoption name Hash value 256
position startpos
go wtime 900000 btime 900000 movestogo 40
```

`OwnBook` defaults to **false**: the internal repertoire exists but every test
this engine has been measured under disables it, and a default nothing is tested
under is not a safe default.

## Strength

Measured at −21.74 ± 25.62 over 400 games against Stash v26 (CCRL blitz 3000) at
20+0.2, so roughly 2980, in August 2026. A further +160 ± 26 has been measured in
self-play since; that figure has **not** been re-anchored against a foreign
opponent, and self-play gains are not strength gains until something outside the
family confirms them.

## License

MIT
