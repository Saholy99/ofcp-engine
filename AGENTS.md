# AGENTS.md

This file contains repository-wide instructions for Codex and any coding agent working in this repo.

## Mission

Build a **correct, testable, deterministic** engine for **heads-up Pineapple Open-Face Chinese Poker (OFC) with Fantasyland**.

The engine comes first. The solver comes later.

Priorities, in order:

1. **Rule correctness**
2. **Deterministic, explicit state transitions**
3. **Test coverage**
4. **Clear interfaces for future solver work**
5. **Performance only after correctness is locked**

---

## Source of Truth

`docs/rules.md` is the canonical rules specification for this project.

When implementing or modifying behavior:

- follow `docs/rules.md` exactly
- do **not** invent or silently assume new game rules
- if code, tests, and `docs/rules.md` disagree, treat `docs/rules.md` as correct unless the user explicitly says otherwise
- if a requested code change implies a rules change, update `docs/rules.md` and tests in the same task

If a rule is ambiguous, do not improvise. Surface the ambiguity clearly in code comments, tests, or the final task summary.

---

## Scope Boundary: Engine vs Solver

Keep the **engine** and the **solver** separate.

### Engine responsibilities
The engine may include:

- card representation
- deck / remaining-card tracking
- board representation
- row evaluation
- legality / foul detection
- action validation
- state transitions
- hidden/public information handling
- scoring
- Fantasyland entry / stay / exit
- full terminal hand resolution

### Solver responsibilities
The solver may include:

- rollout logic
- Monte Carlo evaluation
- MCTS / search
- opponent modeling
- value functions / policies
- caching / transposition strategies for search

### Hard rule
Do **not** place solver logic inside engine modules.

The engine must remain usable as a clean, deterministic simulator independent of any search policy.

---

## Required Design Principles

### 1. Prefer pure functions
Whenever practical, implement core logic as pure functions.

Examples:

- `legal_actions(state)`
- `apply_action(state, action) -> new_state`
- `is_foul(board) -> bool`
- `score_terminal(state) -> result`
- `compare_rows(a, b) -> outcome`

Avoid hidden side effects in rules logic.

### 2. Keep state explicit
State must make hidden information and public information easy to distinguish.

At minimum, game state should make these distinctions explicit:

- visible board cards
- hidden discards
- current private draw
- undealt / remaining cards
- Fantasyland concealed arrangements
- next-hand Fantasyland flags
- button / acting player

Do not hide important game information inside ad hoc helper objects.

### 3. Determinism matters
Use deterministic behavior whenever possible.

- all randomness must be seedable
- do not rely on implicit global randomness
- tests must be reproducible
- if shuffling is used, it must be easy to inject a seed or a prearranged deck order

### 4. Immutability is preferred
Prefer immutable or mostly-immutable state objects for engine transitions.

If mutation is used internally for performance, the public interface should still behave predictably and safely.

### 5. Explicit actions only
Use explicit action types for game progression.

Examples:

- initial 5-card placement
- Pineapple 3-card draw placement + discard
- Fantasyland 13-of-14 set action

Do not bury player decisions inside generic helper functions without an explicit action object or equivalent structured input.

---

## Rules-Specific Invariants That Must Never Be Violated

The implementation must preserve the following invariants from `docs/rules.md`:

1. Top row has capacity 3.
2. Middle row has capacity 5.
3. Bottom row has capacity 5.
4. Once a card is placed, it cannot be moved.
5. A row cannot exceed capacity.
6. In a normal Pineapple draw turn, the player receives exactly 3 cards, places exactly 2, and discards exactly 1.
7. Discarded Pineapple cards are dead and remain hidden from the opponent forever.
8. Fantasyland hands remain concealed until showdown.
9. A hand is legal only if Bottom >= Middle >= Top.
10. A fouled hand receives no royalties.
11. A fouled hand cannot enter Fantasyland.
12. A fouled Fantasyland hand cannot stay in Fantasyland.
13. Fantasyland entry is based on a **legal** hand with **QQ+ on top**.
14. Fantasyland stay conditions must exactly match `docs/rules.md`.
15. If one player fouls, the non-fouling player is scored exactly as a sweep plus their royalties.
16. If both players foul, both score 0.
17. Row ties score 0.
18. The button does not move during a Fantasyland continuation hand.
19. After the Fantasyland continuation hand resolves, normal button rotation resumes.

When touching scoring, legality, Fantasyland, or transitions, preserve these invariants exactly.

---

## Code Organization Expectations

Prefer a structure close to this:

```text
src/ofc/
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
```

Recommended responsibilities:

- `config.py`: variant configuration and constants
- `cards.py`: rank/suit/card types and utilities
- `deck.py`: deck construction, shuffling, drawing, remaining-card accounting
- `board.py`: row/board structures and placement rules
- `evaluator.py`: row ranking, comparison, foul-order comparison helpers
- `actions.py`: structured action definitions
- `state.py`: complete game-state models
- `transitions.py`: legal state progression and action application
- `scoring.py`: row scoring, sweep handling, royalties, terminal scoring
- `fantasyland.py`: enter / stay / exit logic
- `engine.py`: orchestration helpers and public engine APIs

This structure may be adjusted if there is a strong reason, but keep the separation of concerns.

---

## Testing Requirements

Tests are mandatory.

### General rules
- every new public function should have tests
- every bug fix should include a regression test
- changes to scoring, foul detection, row comparison, or Fantasyland logic must include targeted tests
- avoid relying only on integration tests; use focused unit tests too

### Minimum test categories
Maintain tests for:

1. **card / deck correctness**
2. **5-card evaluation**
3. **3-card top-row evaluation**
4. **cross-row foul comparison**
5. **legal / illegal boards**
6. **action validation**
7. **state transitions**
8. **terminal scoring**
9. **royalty lookup**
10. **Fantasyland entry / stay / exit**
11. **full-hand golden scenarios**

### Golden scenarios
Maintain several end-to-end scenarios with fully specified expected outcomes, including:

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
- concealed Fantasyland reveal / showdown behavior

When a bug is discovered, add a golden test if the bug affected end-to-end behavior.

---

## Implementation Style

Use:

- Python 3.12+
- standard library unless the user explicitly approves dependencies
- `dataclasses`, `Enum`, and type hints where useful
- clear names over cleverness
- short functions with explicit invariants
- docstrings on public interfaces
- comments for tricky rule logic, especially scoring and cross-row comparisons

Avoid:

- unnecessary inheritance
- mixing business logic with I/O
- hidden global state
- magic numbers when a named constant belongs in config
- premature optimization that makes rules harder to audit

---

## How to Work on Tasks in This Repo

For non-trivial tasks:

1. read `docs/rules.md`
2. inspect relevant existing modules and tests
3. make a short plan before editing
4. implement the smallest correct change set
5. add or update tests
6. run the relevant tests
7. summarize what changed and any rule-sensitive assumptions

For larger tasks spanning multiple files, prefer incremental commits/changes rather than one monolithic rewrite.

---

## When Editing Rules-Sensitive Code

The following areas are considered high-risk:

- `evaluator.py`
- `scoring.py`
- `fantasyland.py`
- `transitions.py`
- any legality / foul detection logic

When changing one of these areas:

- preserve backward compatibility unless the rules spec changed
- add tests before or with the implementation
- state clearly whether behavior changed intentionally or only the implementation changed

---

## Performance Guidance

Correctness comes first.

Only optimize after correctness and tests are solid.

If performance work is needed:

- profile first
- preserve public interfaces if possible
- do not make core rule logic opaque just for speed
- isolate optimizations so they do not reduce auditability

This is especially important because a future solver will depend on the engine being trustworthy.

---

## Solver Readiness Requirements

Even before solver code exists, engine design should support future search.

That means:

- state cloning / branching should be straightforward
- hidden information should be explicit
- action generation should be deterministic and testable
- terminal scoring should be a pure function
- seeded randomness should be easy to control
- public APIs should be stable enough for rollout and search code later

Do not implement the solver yet unless explicitly asked. Just keep the engine solver-friendly.

---

## Definition of Done

A task is not complete unless:

- the implementation matches `docs/rules.md`
- affected tests are added or updated
- relevant tests pass
- no engine/solver boundary was violated
- no silent rules were invented
- the final summary clearly states any assumptions or unresolved ambiguities

---

## Default Behavior on Ambiguity

If something is unclear:

- prefer the explicit wording in `docs/rules.md`
- do not silently choose a new house rule
- surface the ambiguity clearly
- keep changes minimal until clarified

Correctness and auditability are more important than speed.
