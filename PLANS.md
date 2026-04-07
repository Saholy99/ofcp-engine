# PLANS.md

## Build Plan for the OFC Engine

### Summary
- Ground the implementation strictly in [AGENTS.md](/Users/sahilparikh/Desktop/ofcp-engine/AGENTS.md) and [docs/rules.md](/Users/sahilparikh/Desktop/ofcp-engine/docs/rules.md). The repo currently contains only docs; [docs/scoring_examples.md](/Users/sahilparikh/Desktop/ofcp-engine/docs/scoring_examples.md) is empty.
- Build the engine from deterministic primitives upward: cards/deck, row evaluation, board legality, scoring/Fantasyland pure rules, then explicit state/actions/transitions, then a thin engine facade.
- Keep solver concerns out of the engine. Every rules-sensitive behavior should be implemented as a pure function first, then composed into state transitions.

### Proposed Repo Structure
```text
pyproject.toml
src/ofc/
  __init__.py
  config.py
  cards.py
  deck.py
  board.py
  evaluator.py
  actions.py
  state.py
  transitions.py
  scoring.py
  fantasyland.py
  engine.py
tests/
  test_cards.py
  test_deck.py
  test_evaluator_five_card.py
  test_evaluator_top_row.py
  test_board_legality.py
  test_actions.py
  test_transitions_normal.py
  test_transitions_fantasyland.py
  test_scoring.py
  test_golden_hands.py
```

### Core Models and Public Interfaces
- `cards.py`
  - `Suit`, `Rank`, `Card`
  - Pure helpers: `parse_card()`, `format_card()`, `full_deck()`
- `config.py`
  - Row capacities, royalty tables, player count, Fantasyland thresholds
  - `VariantConfig` for future-proofing without solver logic
- `deck.py`
  - `DeckState` with explicit undealt order
  - Pure helpers: `make_deck(seed | preset_order)`, `draw_n(deck, n) -> (cards, new_deck)`
- `board.py`
  - `RowName`, `Board(top, middle, bottom)`
  - Pure helpers: `place_cards()`, `row_capacity_remaining()`, `board_full()`
- `evaluator.py`
  - `HandCategory`
  - `TopRowValue`, `FiveCardValue`, `ComparableRowValue`
  - Pure helpers: `evaluate_top_row()`, `evaluate_five_card_row()`, `compare_same_size_rows()`, `compare_cross_rows_for_foul()`
- `actions.py`
  - Explicit action types only:
    - `PlaceInitialFiveAction`
    - `PlaceDrawAction`
    - `SetFantasylandHandAction`
- `state.py`
  - `PlayerId`, `HandPhase`, `PlayerState`, `GameState`
  - `PlayerState` should keep visible board, hidden discards, current private draw, Fantasyland flags, and concealed Fantasyland arrangement explicit
  - `GameState` should keep acting player, button, deck, continuation-hand flag, and next-hand Fantasyland status explicit
- `scoring.py`
  - `RowOutcome`, `PlayerScoreBreakdown`, `TerminalResult`
  - Pure helpers: `is_foul()`, `royalties_for_board()`, `score_rows()`, `score_terminal()`
- `fantasyland.py`
  - Pure helpers: `qualifies_for_fantasyland()`, `qualifies_to_stay_in_fantasyland()`, `resolve_next_hand_fantasyland_flags()`
- `transitions.py`
  - Pure helpers: `legal_actions(state)`, `validate_action(state, action)`, `apply_action(state, action)`, `advance_after_showdown(state, result, next_deck)`
- `engine.py`
  - Thin orchestration only: `new_match()`, `new_hand()`, `apply()`, `showdown()`

### Build Order and Milestones
1. **Project scaffold**
   - Create `pyproject.toml`, `src/ofc`, `tests`
   - Use standard-library `unittest` unless dependency approval is given later
2. **Cards and deck**
   - Implement immutable card types and deterministic deck construction/drawing
   - Lock seeded and preset-order behavior with tests
3. **Row evaluation**
   - Implement 5-card evaluation, top-row evaluation, tie-breaks, wheel straight handling, and cross-row comparison for foul checks
4. **Board legality**
   - Implement row capacities, placement validation, board completion checks, and foul detection
5. **Scoring pure rules**
   - Implement royalties, row scoring, sweep logic, foul scoring, and score breakdown objects
6. **Fantasyland pure rules**
   - Implement entry, stay, exit, concealment, and button-continuation rules as pure functions
7. **State and actions**
   - Implement explicit immutable-ish game state and action models for initial placement, draw turns, and Fantasyland set turns
8. **Transitions**
   - Implement deterministic normal-hand flow first, then Fantasyland continuation flow
   - Keep private draws, dead discards, and concealed Fantasyland boards explicit in state
9. **Engine facade and golden tests**
   - Add thin public APIs over transitions/scoring
   - Finish end-to-end fixtures covering all required scenarios

### Highest-Risk Rule Areas
- Cross-row foul comparison, especially equal-category comparisons between 3-card top and 5-card middle
- One-player-foul scoring and royalty treatment
- Fantasyland entry/stay/exit conditions
- Fantasyland continuation hand sequencing and button freeze/resume behavior
- Hidden-information handling:
  - private 3-card draws
  - permanently hidden dead discards
  - concealed Fantasyland arrangements until showdown

### Testing Strategy
- Unit tests for every public pure function
- Focused evaluator tests:
  - all 5-card categories
  - top-row high card/pair/trips only
  - tie-breakers
  - wheel straight / straight flush
  - cross-row foul comparisons with same-category kicker edge cases
- Board/state tests:
  - row capacity enforcement
  - once-placed-cannot-move invariant
  - draw action must place exactly 2 and discard exactly 1
  - no card duplication across boards, discards, private draw, undealt deck
- Scoring tests:
  - royalties per row table
  - row ties score 0
  - sweep bonus
  - one-foul and both-foul outcomes
- Fantasyland tests:
  - legal QQ+ entry
  - fouled hand cannot enter
  - stay conditions by row
  - fouled Fantasyland hand cannot stay
  - button freeze during continuation and resume afterward
- Golden end-to-end tests:
  - legal hand with no royalties
  - legal hand with royalties
  - row ties
  - sweep without royalties
  - sweep with royalties
  - one-player foul
  - both-player foul
  - Fantasyland entry
  - Fantasyland stay
  - Fantasyland exit
  - concealed Fantasyland reveal at showdown
  - one-player Fantasyland continuation hand
  - both-player Fantasyland continuation hand

### Assumptions and Ambiguities
- Resolved in [docs/rules.md](/Users/sahilparikh/Desktop/ofcp-engine/docs/rules.md): terminal scoring is zero-sum. For one-player-foul hands, the legal player’s net result is `+6 + their royalties`, and the fouling player gets the exact negative.
- Fantasyland continuation sequencing will reuse standard hand turn order. The button remains unchanged, the player to the left of the button acts first, and a Fantasyland set action occurs on that player’s turn while remaining concealed until showdown.
- Because [docs/scoring_examples.md](/Users/sahilparikh/Desktop/ofcp-engine/docs/scoring_examples.md) is empty, all implementation details should be validated only against [docs/rules.md](/Users/sahilparikh/Desktop/ofcp-engine/docs/rules.md) and tests created from it.
- For cross-row foul comparison when categories match and the shared ranking prefix is equal, treat the 3-card top row as exhausted once it runs out of kickers. The 5-card row remains stronger if it still has remaining lexicographic kicker detail. This should be documented in tests because the rules imply it but do not give a full equal-prefix example.

### Recommended First Implementation Task
- Create the package/test scaffold and implement only the deterministic primitives:
  - `config.py`
  - `cards.py`
  - `deck.py`
  - `tests/test_cards.py`
  - `tests/test_deck.py`
- Acceptance criteria for that first task:
  - full 52-card deck generation is correct
  - parsing/formatting is stable
  - deck order is reproducible from a seed or preset order
  - drawing removes cards deterministically without duplication
  - no engine state, solver logic, or rules-sensitive transition code is introduced yet
