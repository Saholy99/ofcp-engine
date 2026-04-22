# Solver Improvement Log

This document records solver-focused improvements, why they were added, how to
run them, and what the latest benchmark results showed. The engine remains the
source of truth for rules and state transitions; solver improvements live in
`src/ofc_solver/` and analysis/benchmark support lives in `src/ofc_analysis/`.

## Baseline: Random Rollout Monte Carlo

The original solver ranked each legal root action by applying that action,
running Monte Carlo rollouts with uniformly random legal future actions, and
ranking actions by average zero-sum score.

This was useful as a baseline, but diagnostics showed a major artifact:
rollout values were heavily driven by random future fouls rather than coherent
OFC placement quality.

## Benchmark And Diagnostic Harness

Added a benchmark manifest and CLI workflow:

```bash
python3 -m ofc_analysis.cli benchmark-solver scenarios/benchmarks/solver_diagnostics.json --policy random --json
python3 -m ofc_solver.benchmark_corpus
python3 -m ofc_analysis.cli benchmark-solver scenarios/benchmarks/solver_expansive.json --policy random --json
```

The harness records:

- ranked root actions
- current-hand EV
- continuation-hand EV
- continuation frequency
- root, opponent, and both-foul rates
- root and opponent Fantasyland frequency
- rollout policy decision count
- exact late-search activation frequency and node counts
- labeled top-1/top-3 agreement when a benchmark case has an oracle
- elapsed runtime
- comparison slices by benchmark tag

Generated expansive benchmark scenarios are intentionally ignored by git.

## Heuristic Rollout Policy V1

Added `HeuristicRolloutPolicy` beside the original `RandomRolloutPolicy`.
Random remains the default baseline.

The heuristic policy:

- uses engine `legal_actions()` and `apply_action()` only
- greedily scores legal rollout actions
- uses seeded RNG only for exact score ties
- penalizes completed fouls and dangerous completed row ordering
- rewards legal royalties and Fantasyland entry/stay
- prefers stronger bottom and middle construction before top strength
- penalizes fragile top-heavy shapes
- scores discards based on missed pair, flush, and straight value
- uses bounded Fantasyland candidate generation instead of random
  14-card assignment

Commands:

```bash
python3 -m ofc_analysis.cli solve-move scenarios/regression/immediate_scoring.json --observer player_0 --rollouts 100 --seed 123 --policy heuristic
python3 -m ofc_analysis.cli benchmark-solver scenarios/benchmarks/solver_diagnostics.json --policy heuristic --json
```

## Heuristic Rollout Policy V1.1

The first follow-up improvement made final rollout decisions more exact:

- if a candidate rollout action immediately reaches showdown, score it by the
  actual terminal hand result for the acting player
- keep board-shape scoring for non-terminal rollout actions
- cache repeated solver-layer row evaluations and Fantasyland option rankings
  without changing engine APIs or rule logic

This keeps the policy interpretable while avoiding heuristic approximation in
positions where the engine can already score the result exactly.

## Heuristic Rollout Policy V1.2

The next improvement added bounded exact late-street search:

- when a `DRAW` state has a small remaining normal-action tree, enumerate that
  tree to showdown
- score actions by minimax terminal value from the acting player's perspective
- treat the opponent as choosing the response that minimizes the acting
  player's zero-sum terminal score
- keep the default exact search bounded to at most 3 remaining normal draw
  decisions and 100 enumerated tree nodes
- fall back to the existing heuristic when the remaining tree is too large or
  the state is not a normal draw state

This targets the observed "too greedy" behavior in late streets. The rollout
policy can now look through a small number of forced future decisions instead
of overvaluing a shape that gets punished by the opponent's final response.

## Benchmark Comparison

Added side-by-side benchmark comparison:

```bash
python3 -m ofc_analysis.cli compare-benchmarks /tmp/ofcp_random_expansive.json /tmp/ofcp_heuristic_expansive.json --json
```

The comparison reports aggregate deltas and top-action changes by case.

The comparison now separates two kinds of rates:

- all-action rates, which average diagnostics across every legal root action
- top-action rates, which only average the action each solver run actually
  ranked first in each benchmark case

This distinction matters because all-action root foul rate can be high even
when the chosen top actions are less foul-prone. It also exposes whether exact
late-street search activated during rollout decisions instead of guessing from
aggregate foul rates alone.

The comparison also includes tag slices, so generated final-draw stress cases
can be separated from initial, early, mid, late, Fantasyland, strategy, and
survivability cases.

## Diagnostic Finding: Root Foul Rate Interpretation

After adding top-action and exact-search diagnostics, the high root/both foul
rates should not be interpreted as a single failure mode.

The current evidence points to three separate causes:

- all-action aggregate root foul rate is inflated by bad legal root actions the
  solver did not choose
- many generated final-draw random-walk cases are already structurally doomed,
  so no late-street exact search can rescue them
- non-final top-action root foul rate is still meaningfully above zero, which
  means early greedy placement quality still needs work

## Heuristic Rollout Policy V1.3

V1.3 added a survivability component for incomplete boards:

- penalize unsupported top `QQ+` pairs
- penalize unsupported top trips more heavily
- penalize middle strength that outpaces bottom support
- penalize underbuilt bottom rows in early/mid shapes
- attach debug reasons such as `unsupported-top-pair`,
  `unsupported-top-trips`, `middle-over-bottom-pressure`, and
  `bottom-underbuilt`

It also added strategy/survivability generated benchmark cases and tag-sliced
benchmark comparison output.

## Latest Measured Result

On the generated expansive corpus with 56 cases:

| Metric | Random | Heuristic V1.3 | Delta |
| --- | ---: | ---: | ---: |
| Root foul rate | 0.6961 | 0.4269 | -0.2692 |
| Both-foul rate | 0.4771 | 0.1672 | -0.3099 |
| Top-action root foul rate | 0.3789 | 0.3230 | -0.0559 |
| Top-action both-foul rate | 0.3106 | 0.2360 | -0.0745 |
| Top-action continuation frequency | 0.0124 | 0.0683 | +0.0559 |
| Top-action exact-search activation | 0.0000 | 0.8696 | +0.8696 |
| Labeled top-1 agreement | 1.0000 | 1.0000 | 0.0000 |
| Labeled top-3 agreement | 1.0000 | 1.0000 | 0.0000 |
| Runtime | 7.18s | 124.46s | +117.28s |

Selected tag slices from the same run:

| Slice | Cases | Random Top Root Foul | Heuristic Top Root Foul | Random Top Both-Foul | Heuristic Top Both-Foul |
| --- | ---: | ---: | ---: | ---: | ---: |
| initial_deal | 10 | 0.0000 | 0.2143 | 0.0000 | 0.0714 |
| early_draw | 15 | 0.3902 | 0.1707 | 0.3902 | 0.0976 |
| mid_draw | 11 | 0.1515 | 0.1212 | 0.1212 | 0.1212 |
| late_draw | 8 | 0.4167 | 0.2083 | 0.1667 | 0.0833 |
| final_draw | 12 | 0.8571 | 0.8571 | 0.7429 | 0.7429 |
| strategy | 3 | 0.8889 | 0.5556 | 0.8889 | 0.3333 |

For heuristic top actions, excluding cases tagged `final_draw` gives root foul
rate 0.1746 and both-foul rate 0.0952. Excluding the newly added strategy
stress cases as well gives root foul rate 0.1453 and both-foul rate 0.0769.
Initial+early non-strategy cases improved from the previous 0.1667 root foul
baseline to 0.1333.

Interpretation:

- The heuristic materially reduced all-action foul-heavy rollout artifacts.
- Exact terminal scoring improved late rollout correctness where no heuristic
  approximation is needed.
- Bounded exact late-street search further reduced root and both-foul rates,
  but with a meaningful runtime increase.
- Exact late-search activates frequently on heuristic top actions, so the
  remaining root foul rate is not explained only by exact search failing to
  trigger.
- V1.3 helped early, mid, late, and strategy slices, but initial-deal top
  actions are still noisy and need direct root-action risk handling or more
  rollouts.
- The current bottleneck is still heuristic runtime, especially repeated action
  scoring through engine application during rollouts.
- The next solver-policy work should address root-action risk directly rather
  than relying only on rollout policy improvements after a root action is
  already fixed.
