# patzer

A UCI chess engine written from scratch in Rust — NNUE evaluation on top of a
classical alpha-beta search. Plays live on Lichess as
[@hubilipski](https://lichess.org/@/hubilipski).

## Design

**Search** (`src/search.rs`)
- Iterative deepening with aspiration windows
- Principal-variation search (fail-soft negamax)
- Transposition table — power-of-two buckets, depth/age replacement (`src/tt.rs`)
- Quiescence search with static-exchange-evaluation (SEE) pruning
- Null-move pruning, late-move reductions, futility pruning, check extensions
- Move ordering: TT move → SEE-ranked captures → killers → counter-move →
  butterfly history + continuation history

**Evaluation**
- NNUE (`src/nnue.rs`): efficiently-updatable accumulator, quantized integer
  inference, trained by self-play distillation
- Falls back to a tapered hand-crafted PeSTO evaluation (`src/pesto.rs`) when no
  net is loaded

**Move generation**: [`cozy-chess`](https://crates.io/crates/cozy-chess), perft-verified.

**Development discipline**: search and evaluation changes are accepted only via
equal-time SPRT self-play with pre-registered Elo bounds — never fixed-nodes or
fixed-depth comparisons.

## Build

```sh
cargo build --release
```

## Run (UCI)

```sh
./target/release/patzer
```

```
uci
setoption name EvalFile value path/to/net.nnue   # optional — PeSTO eval used if omitted
setoption name Hash value 128
position startpos
go movetime 1000
```

## Strength

Roughly 2700 Elo by internal calibration against Stockfish `UCI_Elo` brackets.
Cross-version improvements were each validated by equal-time SPRT self-play; the
absolute figure carries the usual caveats of engine-vs-engine Elo estimates.

## License

MIT
