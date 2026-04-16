# SOLVER_PLAN.md

## Summary
- Build a **narrow analysis harness first** in `src/ofc_analysis/`, then build a **baseline Monte Carlo move-ranker** in `src/ofc_solver/`.
- Keep `src/ofc/` solver-free. Reuse existing engine APIs for action generation, transitions, showdown, scoring, and continuation-hand flow instead of reimplementing any rules.
- First persisted scenario format is **JSON exact state** using the current engine dataclasses and `DEFAULT_CONFIG` only.
- First CLI is **text-first with `--json` output**, and first solver scope is:
  - root phases: `INITIAL_DEAL` and `DRAW` only
  - objective: rank the acting player’s current legal moves
  - horizon: current hand + at most one immediate Fantasyland continuation hand

## Module Plan
### `src/ofc_analysis/`
- `__init__.py`
  - Re-export the public analysis helpers.
- `scenario.py`
  - Define the exact-state scenario schema and loader.
  - Convert JSON into real engine objects: `GameState`, `PlayerState`, `Board`, `DeckState`, and engine action/row/card types where needed.
  - Validate card conservation and required field consistency.
- `action_codec.py`
  - Canonical action rendering for CLI and tests.
  - Provide deterministic action payloads with `type`, placements, discard, and stable action index.
- `observation.py`
  - Project an exact `GameState` into a minimal observer-facing `PlayerObservation`.
  - Redact opponent hidden data while preserving the observer’s own private information.
- `render.py`
  - Human-readable and JSON renderers for exact state, observation, legal actions, and solver results.
- `cli.py`
  - Minimal command-line interface for state inspection, action listing, and solver invocation.

### `src/ofc_solver/`
- `__init__.py`
  - Re-export the public solver entrypoints.
- `models.py`
  - `MoveEstimate`, `MoveAnalysis`, and any small enums/constants for solver output.
- `sampler.py`
  - Sample a full hidden state consistent with a `PlayerObservation`.
  - Produce a complete `GameState` plus RNG-driven next-hand deck sampling for continuation-hand rollouts.
- `rollout_policy.py`
  - Define the rollout-policy interface.
  - Implement a baseline random rollout policy.
  - Include a dedicated sampled Fantasyland-set action generator for rollout-only use.
- `rollout.py`
  - Drive one rollout from a root state after a chosen action.
  - Finish the current hand via engine transitions and, if triggered, exactly one continuation hand.
- `monte_carlo.py`
  - Rank root legal actions by repeated rollouts.
  - Provide exact-state debug wrappers and observation-based solver entrypoints.

## Build Order
1. **Scenario loader**
   - Add `src/ofc_analysis/scenario.py`.
   - Acceptance: load a JSON exact state into a valid engine `GameState` or fail with clear validation errors.
2. **Action codec + renderers**
   - Add `src/ofc_analysis/action_codec.py` and `src/ofc_analysis/render.py`.
   - Acceptance: exact state and legal actions render deterministically in text and JSON.
3. **Observation model**
   - Add `src/ofc_analysis/observation.py`.
   - Acceptance: same exact state can be projected for `player_0` and `player_1` with correct redaction.
4. **CLI harness**
   - Add `src/ofc_analysis/cli.py`.
   - Acceptance: `show-state`, `list-actions`, and `solve-move` command surfaces are wired, even if `solve-move` is stubbed initially.
5. **Solver output models + rollout policy**
   - Add `src/ofc_solver/models.py` and `src/ofc_solver/rollout_policy.py`.
   - Acceptance: a baseline random policy can choose a legal action in all rollout phases, including sampled Fantasyland-set actions.
6. **Exact-state rollout core**
   - Add `src/ofc_solver/rollout.py`.
   - Acceptance: from an exact state and root action, the solver can simulate to terminal scoring and, when needed, one continuation hand.
7. **Infoset sampler**
   - Add `src/ofc_solver/sampler.py`.
   - Acceptance: sample complete hidden states consistent with an observation without violating card conservation or public facts.
8. **Monte Carlo ranker**
   - Add `src/ofc_solver/monte_carlo.py`.
   - Acceptance: rank current legal root actions for `INITIAL_DEAL` and `DRAW` using seeded rollouts.
9. **CLI solver integration**
   - Finish `solve-move` in `src/ofc_analysis/cli.py`.
   - Acceptance: a scenario file can be solved from the CLI with readable rankings and JSON output.

## Existing Engine APIs To Reuse
- `ofc.transitions.legal_actions(state)`
  - Root action enumeration for `INITIAL_DEAL` and `DRAW`.
- `ofc.transitions.apply_action(state, action)`
  - Apply root actions and rollout actions.
- `ofc.engine.showdown(state)`
  - Resolve terminal current-hand scoring and next-hand Fantasyland flags.
- `ofc.transitions.advance_after_showdown(state, result, next_deck)`
  - Enter the immediate Fantasyland continuation hand when triggered.
- `ofc.state.get_player(...)`
  - Access acting-player and observer-specific private information.
- `ofc.state.effective_board(...)`
  - Reveal concealed Fantasyland boards only when appropriate.
- `ofc.state.all_known_cards(...)`
  - Scenario validation and sampler consistency checks.
- `ofc.actions.PlaceInitialFiveAction`, `PlaceDrawAction`, `SetFantasylandHandAction`, `Placement`
  - Reuse engine action types directly in solver/harness code.
- `ofc.cards.parse_card`, `format_card`
  - Scenario parsing and CLI rendering.
- `ofc.deck.DeckState`
  - Exact-state scenario loading and sampled continuation-hand deck creation.

Use these directly. Do not duplicate legality, evaluation, foul detection, scoring, or continuation-hand logic inside analysis/solver packages.

## Minimal Scenario Format First
Use **JSON exact state** only. No partial-observation files in v1.

### Top-level shape
- `version`
  - Start with `"1"`.
- `state`
  - `hand_number`
  - `button`
  - `acting_player`
  - `phase`
  - `is_continuation_hand`
  - `next_hand_fantasyland`
  - `deck`
    - `undealt_cards`
  - `players`
    - exactly two entries

### Per-player shape
- `player_id`
- `board`
  - `top`
  - `middle`
  - `bottom`
- `hidden_discards`
- `current_private_draw`
- `fantasyland_active`
- `concealed_fantasyland_board`
  - nullable, same row shape as `board`
- `concealed_fantasyland_discard`
  - nullable
- `initial_placement_done`
- `normal_draws_taken`
- `fantasyland_set_done`

### Validation rules
- Use `DEFAULT_CONFIG` only in v1; do not add config-overrides to the scenario schema yet.
- Reject duplicates across all card-holding locations.
- Require card conservation across visible boards, hidden discards, private draws, concealed Fantasyland data, and undealt deck.
- Allow any phase to be loaded for inspection.
- Restrict `solve-move` to exact states whose root phase is `INITIAL_DEAL` or `DRAW`.

## Minimal Observation Model First
Create `PlayerObservation` in `src/ofc_analysis/observation.py`.

### Required fields
- `observer`
- `acting_player`
- `phase`
- `hand_number`
- `button`
- `is_continuation_hand`
- `next_hand_fantasyland`
- `public_boards`
  - visible board rows for both players only
- `own_private_draw`
- `own_hidden_discards`
- `own_fantasyland_active`
- `opponent_fantasyland_active`
- `opponent_hidden_discard_count`
- `unseen_card_count`

### Explicit exclusions
- Do not expose opponent hidden discards.
- Do not expose undealt deck order.
- Do not expose opponent private draw.
- Do not expose opponent concealed Fantasyland arrangement.

This observation model is enough for the solver MVP and the CLI inspection commands.

## Minimal CLI Commands First
Implement these commands in `src/ofc_analysis/cli.py`.

- `show-state <scenario.json> [--observer player_0|player_1] [--json]`
  - Without `--observer`, show exact state.
  - With `--observer`, show the projected observation.
- `list-actions <scenario.json> [--json]`
  - List deterministic root legal actions with stable indices and canonical action payloads.
  - For v1, support only root phases `INITIAL_DEAL` and `DRAW`; reject `FANTASYLAND_SET` to avoid million-action dumps.
- `solve-move <scenario.json> --observer player_0|player_1 --rollouts N --seed S [--json]`
  - Rank the acting player’s legal root actions from that player’s information set.
  - Require `observer == acting_player` in v1.

Do not add `apply-action` or a web UI in this phase.

## Minimal Solver Interfaces First
### Output models
- `MoveEstimate`
  - `action_index`
  - `action`
  - `mean_value`
  - `stddev`
  - `sample_count`
  - `min_value`
  - `max_value`
- `MoveAnalysis`
  - `observer`
  - `phase`
  - `rollouts_per_action`
  - `rng_seed`
  - `ranked_actions`

### Public entrypoints
- `rank_actions_from_state(state, *, observer, rollouts_per_action, rng_seed) -> MoveAnalysis`
  - Exact-state debug entrypoint.
- `rank_actions_from_observation(observation, *, exact_state_loader, rollouts_per_action, rng_seed) -> MoveAnalysis`
  - Real solver-facing entrypoint using infoset sampling.

### Rollout behavior
- Enumerate root actions using `ofc.transitions.legal_actions`.
- For each root action:
  - sample a full hidden state consistent with the observation
  - apply the fixed root action
  - simulate the rest of the current hand with baseline rollout policies for both players
  - resolve showdown
  - if the result triggers a continuation hand, sample the next-hand deck and simulate exactly one continuation hand
  - return total zero-sum value from the root player’s perspective

### Baseline rollout policy
- Normal phases:
  - sample uniformly from engine `legal_actions(state)`
- Fantasyland-set rollout phase:
  - do **not** enumerate the full action space
  - sample one discard and a random row assignment for the remaining 13 cards that respects row capacities only
  - do **not** reject fouled final boards, because fouls are legal outcomes

## Highest-Risk Areas
- **Infoset sampling correctness**
  - dead hidden discards, current private draws, public boards, and undealt cards must stay consistent
- **Continuation-hand valuation**
  - the solver must include exactly one immediate Fantasyland continuation hand and stop there
- **Fantasyland rollout action generation**
  - exhaustive legal-action enumeration is too large for rollout use
- **Deterministic reproducibility**
  - one RNG seed must control hidden-state sampling, rollout action selection, and continuation-hand deck sampling
- **Action identity stability**
  - `list-actions` and `solve-move` must refer to the same canonical action encoding
- **Observation leakage**
  - observer projections must never expose opponent hidden cards or deck order
- **Engine/solver boundary**
  - no rollout logic, sampling logic, or policy logic should be added under `src/ofc/`

## Test Plan
### Phase 1: scenario loader
Add:
- `tests/test_analysis_scenario.py`

Cover:
- valid exact-state JSON loads into engine objects
- duplicate-card rejection
- card-conservation rejection
- unsupported enum/value rejection
- loading supported non-solver phases for inspection

### Phase 2: action codec + renderers
Add:
- `tests/test_analysis_action_codec.py`
- `tests/test_analysis_render.py`

Cover:
- stable text rendering
- stable JSON rendering
- deterministic action ordering and encoding
- round-trip rendering for placements and discards

### Phase 3: observation model
Add:
- `tests/test_analysis_observation.py`

Cover:
- observer sees own private draw and own hidden discards
- opponent hidden discards are redacted to counts
- visible boards are preserved exactly
- concealed Fantasyland information stays hidden from the opponent

### Phase 4: CLI harness
Add:
- `tests/test_analysis_cli.py`

Cover:
- `show-state` exact mode
- `show-state --observer`
- `list-actions` happy path
- `list-actions` rejects unsupported root phases
- `--json` output shape is stable

### Phase 5: rollout policy and exact-state rollout core
Add:
- `tests/test_solver_rollout_policy.py`
- `tests/test_solver_rollout.py`

Cover:
- random normal-phase rollout actions are engine-legal
- sampled Fantasyland-set rollout action respects capacities and uses all 14 cards exactly once
- rollout reaches terminal value deterministically with a fixed seed
- rollout includes one continuation hand when triggered
- rollout stops after that continuation hand

### Phase 6: infoset sampler
Add:
- `tests/test_solver_sampler.py`

Cover:
- sampled exact state matches all public facts
- sampled exact state preserves observer-known private facts
- deck/discard/private-draw partition remains valid
- no card duplication across sampled locations
- if no hidden information remains, repeated sampling returns the exact same state

### Phase 7: Monte Carlo ranker
Add:
- `tests/test_solver_monte_carlo.py`

Cover:
- `rank_actions_from_state` returns ranked root actions for `INITIAL_DEAL`
- same for `DRAW`
- unsupported root phases are rejected clearly
- fixed seed gives repeatable results
- observation-mode and exact-state mode agree when no hidden information remains
- a canned Fantasyland-entry scenario ranks actions using current-hand plus continuation-hand value, not current-hand value alone

## Assumptions and Defaults
- `SOLVER_PLAN.md` should drive implementation without any engine behavior changes.
- v1 scenarios use `DEFAULT_CONFIG` only.
- v1 CLI is text-first with `--json`.
- v1 solver root phases are `INITIAL_DEAL` and `DRAW` only.
- v1 rollout horizon is current hand plus at most one immediate Fantasyland continuation hand.
- v1 solver is information-set based, but the rollout core also exposes an exact-state debug entrypoint.
- No web UI, notebook-specific helpers, MCTS, opponent modeling, or multi-hand match horizon in this phase.
