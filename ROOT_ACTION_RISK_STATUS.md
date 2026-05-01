# Root Action Risk Status

## 1. Project Summary

This repository contains a deterministic Python implementation of heads-up
Pineapple Open-Face Chinese Poker with Fantasyland. The project is intentionally
split into three layers:

- `src/ofc/` is the engine. It owns cards, decks, board shape, explicit actions,
  state transitions, row evaluation, foul detection, royalties, Fantasyland
  entry/stay/exit logic, terminal scoring, and hand advancement.
- `src/ofc_analysis/` is the inspection and tooling layer. It owns exact-state
  scenario loading, player-facing observations, deterministic rendering,
  action encoding, benchmark rendering, and the command-line interface.
- `src/ofc_solver/` is the solver layer. It ranks the acting player's current
  legal root actions by combining engine transitions, information-set sampling,
  rollout policies, diagnostics, and now optional root-action risk scoring.

The source of truth for rules is `docs/rules.md`. The engine remains
solver-free. The solver calls engine APIs such as `legal_actions()`,
`apply_action()`, `showdown()`, and scoring helpers, but solver policy logic does
not live in `src/ofc/`.

The interactive CLI is exposed through `python3 -m ofc_analysis.cli`. It can:

- show exact scenarios with `show-state`
- project observer-facing hidden-information views with `show-state --observer`
- list legal root actions with stable action indices
- run the move ranker with `solve-move`
- run solver benchmark manifests with `benchmark-solver`
- compare two benchmark JSON payloads with `compare-benchmarks`
- play a one-hand interactive workflow with solver suggestions using `play-hand`
- run the focused root-action-risk comparison with `benchmark-root-action-risk`

The project is currently in an engine-complete and solver-diagnostics phase.
The engine already supports the rule set needed for deterministic simulation.
Solver work is still incremental and diagnostic-driven. The current objective is
not a full optimal solver. The current objective is improving the quality and
interpretability of root action rankings without moving solver policy into the
engine or adding a broad expensive search layer too early.

## 2. Solver Stage History

### Baseline: Random Rollout Monte Carlo

The original solver ranked each legal root action by applying that action,
simulating the rest of the current hand with uniformly random legal future
actions, optionally simulating one immediate Fantasyland continuation hand, and
averaging the zero-sum result from the acting player's perspective.

Why it was added:

- It provided the first end-to-end move ranker.
- It exercised the engine through real transitions and terminal scoring.
- It gave a reproducible baseline for scenario and benchmark development.

Problem it tried to solve:

- "Given this root state, which legal action looks best under sampled futures?"

Tradeoff introduced:

- Random future play produces very noisy values.
- Random play fouls often, so root action values can be dominated by rollout
  artifacts instead of meaningful OFC structure.
- It is useful as a baseline, but not strong enough as a policy.

Runtime and behavior:

- Runtime is relatively low.
- Foul behavior is poor because both players frequently make random illegal
  final boards.
- Benchmark foul rates from this stage are best read as a baseline artifact, not
  as a measure of good play.

### Heuristic Rollout Policy V1

V1 introduced `HeuristicRolloutPolicy` beside the original random policy. The
root ranker still enumerated legal root actions with the engine and still used
rollouts. The difference was that future rollout actions were chosen greedily by
a deterministic board-shape heuristic, with seeded random tie-breaking.

What changed:

- Future rollout decisions became heuristic instead of uniformly random.
- The policy rewarded stronger lower-row construction, royalties, legal
  Fantasyland entry, and Fantasyland stay.
- It penalized completed fouls and dangerous completed row ordering.
- It scored discards based on rank, pair-breaking, flush potential, and straight
  potential.
- It sampled Fantasyland set actions through a bounded candidate generator
  rather than enumerating the full combinatorial action space.

Why it was added:

- Random rollout noise was too high.
- The solver needed a cheap policy that played plausible OFC shapes.

Problem it tried to solve:

- Reduce random future foul artifacts and make rollout values more meaningful.

Tradeoff introduced:

- The heuristic is interpretable but not exact.
- It can be greedy: a placement can look good locally while creating future row
  pressure.
- It adds policy computation to every rollout decision.

Runtime and behavior:

- Runtime increased compared with random rollouts.
- All-action and top-action foul rates improved compared with random rollouts.
- Some root action decisions still looked unstable, especially on early streets.

### Heuristic V1.1: Terminal Exact Scoring

V1.1 added exact scoring when a rollout-policy candidate action immediately
reaches showdown.

What changed:

- If applying a candidate action reaches terminal scoring, the rollout policy
  scores that action with the engine's actual terminal result.
- Non-terminal actions still use board-shape heuristic scoring.
- Solver-side caching was added for repeated row-evaluation work.

Why it was added:

- At terminal states, the engine already knows the exact answer.
- Using a heuristic approximation at the final decision can produce avoidable
  errors.

Problem it tried to solve:

- Late street mistakes where the policy ignored exact terminal value.

Tradeoff introduced:

- Minimal conceptual complexity.
- Slightly more engine calls in terminal-adjacent states.

Runtime and behavior:

- Runtime increased modestly where terminal checks are available.
- Final-action quality improved because exact scoring replaced approximation.
- This did not solve early root-action quality because early actions still rely
  on sampled futures.

### Heuristic V1.2: Bounded Exact Late-Street Search

V1.2 added bounded exact search for small late-street normal draw trees.

What changed:

- When a `DRAW` state has a small remaining normal-action tree, the heuristic
  rollout policy enumerates that tree to showdown.
- It scores actions by minimax terminal value from the acting player's
  perspective.
- The default search is bounded by remaining decisions and node count.
- If the tree is too large, the policy falls back to normal heuristic scoring.

Why it was added:

- Some late positions are small enough to solve exactly.
- Greedy rollout choices can fail badly near the end of a hand.

Problem it tried to solve:

- Late-street tactical errors and final-row ordering mistakes.

Tradeoff introduced:

- Runtime can increase sharply when many rollout states trigger exact search.
- The search is intentionally bounded, so it is not a general solver.
- It improves late streets more than initial or early root decisions.

Runtime and behavior:

- Exact-search activation became visible in benchmark diagnostics.
- Late decision quality improved.
- Runtime rose materially in heuristic benchmark runs, especially on generated
  corpora where many rollout states qualify for exact late search.

### Heuristic V1.3: Survivability Scoring

V1.3 added survivability penalties inside the heuristic rollout policy for
incomplete boards.

What changed:

- Penalized unsupported top `QQ+` pairs.
- Penalized unsupported top trips more heavily.
- Penalized middle strength that outpaces bottom support.
- Penalized underbuilt bottom rows.
- Added strategy/survivability generated benchmark cases.
- Added tag-sliced comparison output to separate initial, early, mid, late,
  final, Fantasyland, strategy, and survivability slices.

Why it was added:

- The heuristic could still choose locally attractive shapes that were hard to
  finish legally.
- The main remaining issue was not only terminal scoring. It was incomplete
  board survivability.

Problem it tried to solve:

- Reduce future fouls caused by fragile early and mid-board structures.

Tradeoff introduced:

- More heuristic terms means more tuning surface.
- These terms are still rollout-policy terms, so they influence future rollout
  decisions after a root action is already fixed.
- Broadening heuristics too much can make solver behavior harder to audit.

Runtime and behavior:

- V1.3 improved foul behavior on many non-final slices.
- The improvement log reports a large drop in all-action root foul rate and
  both-foul rate on the generated expansive corpus.
- Runtime remained a concern, especially because exact late search frequently
  activates inside heuristic top-action rollouts.
- Initial-deal and early-root action quality still needed direct attention.

## 3. Benchmark And Diagnostic Metrics

### Root Foul Rate

Root foul rate is the weighted rate at which the root player fouls by the end of
the current hand across all sampled rollouts and all legal root actions in a run.

This metric is useful for understanding the action space, but it can be
misleading by itself. A manifest can contain many legal but obviously bad root
actions. If those bad actions foul, all-action root foul rate rises even if the
solver's top-ranked action is reasonable.

### Top-Action Root Foul Rate

Top-action root foul rate only looks at the action each solver run ranked first
for each case. It answers a more important current question:

> When the solver picks its favorite root action, how often does that action's
> rollout outcome foul?

This matters more than all-action root foul rate for current solver work because
the user sees and may play the top-ranked action.

### Both-Foul Rate

Both-foul rate is the rate at which both players foul in sampled outcomes. High
both-foul rates often indicate rollout-policy weakness, randomly generated
doomed states, or stress cases where both boards are already structurally bad.

### Top-Action Both-Foul Rate

Top-action both-foul rate measures both-foul outcomes only for selected top
actions. It is a better signal for solver recommendation quality than aggregate
both-foul rate.

### Continuation Frequency

Continuation frequency is the rate at which a rollout produces an immediate
Fantasyland continuation hand. It matters because the solver's objective
includes current hand value plus one immediate continuation hand. A low
continuation frequency can mean the policy is not reaching Fantasyland often.
A high continuation frequency can be good, but only if it is not created by
fragile or fouling boards.

### Exact-Search Activation

Exact-search activation measures how often heuristic rollout decisions used the
bounded exact late-street search. The benchmark records both frequency and node
counts.

This metric is important for two reasons:

- It explains runtime growth.
- It separates "the policy guessed" from "the policy solved a bounded late tree."

If exact search activates frequently but top-action foul rate remains high, the
remaining problem is probably earlier root structure, not just missing late
search.

### Labeled Top-1 And Top-3 Agreement

Some benchmark cases have expected action indices. For those cases:

- labeled top-1 agreement means the solver's top action is one of the expected
  actions
- labeled top-3 agreement means at least one expected action appears in the
  solver's top three

These metrics are useful when the expected action is reliable. They are not a
complete quality measure because most generated benchmark cases are unlabeled.

### Runtime

Runtime is elapsed wall-clock time for a benchmark run or case. It matters
because the solver can become unusable if every improvement adds broad search or
large heuristic cost everywhere.

For the current stage, runtime is especially important because V1.2 exact search
already created a meaningful cost increase. Root-action risk was designed as a
cheap scoring adjustment rather than a new broad search.

### Which Metrics Matter Most Right Now

The most important current metrics are:

- top-action root foul rate
- top-action both-foul rate
- top-action continuation frequency
- runtime
- tag-sliced versions of those metrics for `initial_deal`, `early_draw`, and
  non-final cases

All-action rates still matter, but they are secondary because they average over
bad actions the solver did not choose.

## 4. Current Bottleneck

The current bottleneck is root-action quality. The solver can now play future
rollout decisions more coherently than the original random policy, and it can
solve small late-street trees exactly. But if the root action itself creates a
fragile top-heavy shape or underbuilt bottom row, later rollout improvements are
trying to recover after the most important placement choice has already been
fixed.

This is most visible in `INITIAL_DEAL` and early `DRAW` decisions:

- Initial placements define the board's structural direction.
- Early draw placements can lock the top row, overbuild the middle, or starve
  the bottom before there is enough information to repair the shape.
- Rollout noise is still high because many futures remain.

This is why root-action risk is a better next target than immediately adding
beam search everywhere. Beam search could improve some decisions, but it would
also expand runtime and implementation complexity across many states. It would
also risk hiding structural policy mistakes behind search width instead of
making the root ranking more interpretable.

Broad heuristic growth is also risky. Adding many policy terms everywhere makes
the solver harder to audit and tune. Root-action risk is intentionally narrower:
it only adjusts the root ranking, only for `INITIAL_DEAL` and non-final `DRAW`
roots, and it reports named reasons.

## 5. What The Root-Action Risk Module Adds

The new module is `src/ofc_solver/root_action_risk.py`.

It adds:

- `RootRiskComponent`, an interpretable sub-score with a name, contribution, and
  detail string
- `RootActionRiskAssessment`, the aggregate contribution and reason list for one
  candidate root action
- `score_root_action(state, action)`, a cheap deterministic root-only evaluator

The module currently targets these root-risk patterns:

- unsupported top `QQ+` pairs
- unsupported top trips
- middle support that outpaces bottom support
- underbuilt bottom rows
- early top-row slot closure that reduces future flexibility

Where it plugs in:

- `rank_actions_from_state(..., root_action_risk=True)` applies the adjustment
  to root action rankings.
- `rank_actions_from_observation(..., root_action_risk=True)` does the same for
  observation-based solving.
- `solve-move --root-action-risk` exposes the option in the CLI.
- `benchmark-root-action-risk` compares heuristic baseline against
  heuristic-plus-root-risk.
- Benchmark and solve JSON output now include `rollout_mean_value`,
  `root_risk_score`, and `root_risk_reasons` for ranked actions.

What it does not try to solve:

- It does not change engine legality, scoring, foul detection, royalties, or
  Fantasyland rules.
- It does not replace the rollout policy.
- It does not add beam search or MCTS.
- It does not apply during final draw roots.
- It does not attempt to be a complete poker evaluator. It uses engine
  evaluation where needed and only adds solver-side risk heuristics.
- It does not guarantee lower foul rates on every benchmark slice. It creates a
  focused, measurable adjustment so the benchmark can show whether the root
  ranking changed in the intended direction.

## 6. Root-Action-Risk Benchmark Command

The focused command is:

```bash
python3 -m ofc_analysis.cli benchmark-root-action-risk scenarios/benchmarks/solver_diagnostics.json --json
```

By default, it compares:

- left: `heuristic`
- right: `heuristic+root-risk`

The command defaults to the `initial_deal` and `early_draw` tags and excludes
`final_draw`. It can be sliced further:

```bash
python3 -m ofc_analysis.cli benchmark-root-action-risk scenarios/benchmarks/solver_expansive.json --include-tag initial_deal --json
python3 -m ofc_analysis.cli benchmark-root-action-risk scenarios/benchmarks/solver_expansive.json --include-tag early_draw --exclude-strategy --json
python3 -m ofc_analysis.cli benchmark-root-action-risk scenarios/benchmarks/solver_expansive.json --phase draw --non-final --json
```

The JSON output is comparison-oriented. It includes:

- aggregate left/right metrics
- deltas
- tag slices
- top-action changes
- per-case top ranked actions for both sides
- root-risk score and reasons on the right-side ranked actions

The command is intentionally not a new general benchmark framework. It reuses
the existing manifest, case, diagnostics, aggregate, and comparison machinery.

## 7. Recommended Next Steps

1. Run the focused root-action-risk benchmark on the generated expansive corpus:

```bash
python3 -m ofc_solver.benchmark_corpus
python3 -m ofc_analysis.cli benchmark-root-action-risk scenarios/benchmarks/solver_expansive.json --json > /tmp/ofcp_root_action_risk.json
```

2. Compare tag slices before tuning weights:

- `initial_deal`
- `early_draw`
- non-final cases
- non-strategy cases
- strategy/survivability stress cases separately

3. Inspect top-action changes case by case. The first question should be whether
the changed top action is structurally more survivable, not whether one noisy
single-rollout value improved.

4. Add more labeled root-action cases only where the expected action is
defensible. Avoid labeling broad generated cases without a clear oracle.

5. Keep root-risk weights small and interpretable. If large weights become
necessary, that is evidence the rollout values or benchmark case design need
more investigation.

6. Profile before adding broader search. V1.2 already showed that exact search
can dominate runtime. Beam search should wait until root-action diagnostics show
that cheap, local root-risk scoring is not enough.

7. Keep all future solver work solver-side. The engine should remain a clean,
deterministic simulator governed by `docs/rules.md`.
