# OFC Engine

This repo contains a deterministic Python engine for heads-up Pineapple Open-Face Chinese Poker with Fantasyland. The rules live in `docs/rules.md`; engine code stays in `src/ofc/` and is kept separate from analysis and solver code.

## Layout

- `src/ofc/`: core game engine, including cards, boards, actions, transitions, scoring, and Fantasyland rules.
- `src/ofc_analysis/`: scenario loading, player observations, deterministic rendering, and the CLI.
- `src/ofc_solver/`: simple Monte Carlo move-ranker for the acting player's current move.
- `scenarios/regression/`: small canned exact-state JSON scenarios for CLI and solver regression tests.
- `tests/`: unit, integration, analysis, and solver tests.

## Run Tests

```bash
python3 -m unittest discover -s tests
```

## Inspect A Scenario

```bash
python3 -m ofc_analysis.cli show-state scenarios/regression/immediate_scoring.json
python3 -m ofc_analysis.cli list-actions scenarios/regression/immediate_scoring.json
```

## Run The Simple Solver

The current solver is a baseline Monte Carlo ranker. It estimates each legal move by applying that move, simulating many random futures, and ranking moves by average zero-sum score.

```bash
python3 -m ofc_analysis.cli solve-move scenarios/regression/immediate_scoring.json --observer player_0 --rollouts 100 --seed 123
```

More rollouts usually produce less noisy estimates but take longer. The seed makes results reproducible.

## Run Solver Benchmarks

The checked-in benchmark manifest is small and stable. To generate a larger local diagnostic corpus, run the generator first; generated scenarios are ignored by git.

```bash
python3 -m ofc_analysis.cli benchmark-solver scenarios/benchmarks/solver_diagnostics.json --policy random --json
python3 -m ofc_solver.benchmark_corpus
python3 -m ofc_analysis.cli benchmark-solver scenarios/benchmarks/solver_expansive.json --policy random --json
```

## Play One Hand Interactively

Use `play-hand` to play one hand from a single hero seat. Hero turns show the top 3 solver suggestions;
opponent turns only ask for visible placements.

```bash
python3 -m ofc_analysis.cli play-hand --hero player_0 --button player_1 --no-fantasyland --rollouts 5 --seed manual
```

For your own turns, enter all dealt cards and use `top`, `middle`, `bottom`, or `discard` in card order.
To choose a solver suggestion, type its displayed rank, such as `1` for the top suggestion.
To choose a specific legal action after typing `list`, use `action N`.
For opponent draw turns, enter only the 2 visible placed cards and then their rows;
hidden discards are tracked internally as unknown cards.
