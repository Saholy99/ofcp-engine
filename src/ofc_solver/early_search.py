"""Explainable early-game candidate pruning for root solver actions."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ofc.actions import GameAction
from ofc.board import Board, ROW_ORDER, RowName, board_card_count, row_cards
from ofc.cards import Card, Rank
from ofc.evaluator import HandCategory, evaluate_five_card_row, evaluate_top_row
from ofc.state import GameState, HandPhase, get_player
from ofc.transitions import apply_action, legal_actions
from ofc_solver.heuristic_policy import HeuristicRolloutPolicy
from ofc_solver.root_action_risk import RootRiskConfig, score_root_action


@dataclass(frozen=True)
class EarlySearchConfig:
    """Configuration for bounded early root-action candidate pruning."""

    beam_size: int = 48
    candidate_extra_rollouts: int = 0
    draw_safe_candidates: bool = True
    draw_baseline_keep: int = 8
    draw_safety_keep: int = 8

    def __post_init__(self) -> None:
        if self.beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if self.candidate_extra_rollouts < 0:
            raise ValueError("candidate_extra_rollouts must be non-negative")
        if self.draw_baseline_keep < 0:
            raise ValueError("draw_baseline_keep must be non-negative")
        if self.draw_safety_keep < 0:
            raise ValueError("draw_safety_keep must be non-negative")


@dataclass(frozen=True)
class PatternAssessment:
    """Pattern-prior score and interpretable reasons for a card group."""

    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class EarlySearchCandidate:
    """One legal root action plus its deterministic early-game prior."""

    action_index: int
    action: GameAction
    pattern_score: float
    reasons: tuple[str, ...]
    candidate_rank: int
    selection_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EarlySearchCandidateSet:
    """Selected root candidates and the size of the unpruned legal set."""

    candidates: tuple[EarlySearchCandidate, ...]
    total_legal_actions: int

    @property
    def candidate_count(self) -> int:
        return len(self.candidates)

    @property
    def pruning_ratio(self) -> float:
        if self.total_legal_actions == 0:
            return 0.0
        return 1.0 - (len(self.candidates) / self.total_legal_actions)


def select_early_search_candidates(
    state: GameState,
    *,
    config: EarlySearchConfig | None = None,
) -> EarlySearchCandidateSet:
    """Return a deterministic beam of legal root actions for early phases."""

    effective_config = config or EarlySearchConfig()
    if state.phase not in {HandPhase.INITIAL_DEAL, HandPhase.DRAW}:
        raise ValueError("early search supports only initial_deal and draw root phases")

    scored = tuple(
        _score_action(state, action_index, action)
        for action_index, action in enumerate(tuple(legal_actions(state)))
    )
    ranked = tuple(
        sorted(
            scored,
            key=lambda candidate: (-candidate.pattern_score, candidate.action_index),
        )
    )
    if state.phase == HandPhase.DRAW and effective_config.draw_safe_candidates:
        kept = _select_safe_draw_candidates(state, ranked, effective_config)
    else:
        kept = _select_pattern_candidates(ranked, effective_config.beam_size)
    return EarlySearchCandidateSet(
        candidates=tuple(
            EarlySearchCandidate(
                action_index=candidate.action_index,
                action=candidate.action,
                pattern_score=candidate.pattern_score,
                reasons=candidate.reasons,
                candidate_rank=rank,
                selection_reasons=candidate.selection_reasons,
            )
            for rank, candidate in enumerate(kept, start=1)
        ),
        total_legal_actions=len(scored),
    )


def _select_pattern_candidates(
    ranked: tuple[EarlySearchCandidate, ...],
    beam_size: int,
) -> tuple[EarlySearchCandidate, ...]:
    return tuple(
        _add_selection_reason(candidate, "pattern-prior")
        for candidate in ranked[: min(beam_size, len(ranked))]
    )


def _select_safe_draw_candidates(
    state: GameState,
    ranked_by_pattern: tuple[EarlySearchCandidate, ...],
    config: EarlySearchConfig,
) -> tuple[EarlySearchCandidate, ...]:
    candidate_by_action = {candidate.action: candidate for candidate in ranked_by_pattern}
    selected: dict[int, EarlySearchCandidate] = {}
    order: list[int] = []

    def add(candidate: EarlySearchCandidate, reason: str) -> bool:
        existing = selected.get(candidate.action_index)
        if existing is None:
            if len(order) >= config.beam_size:
                return False
            selected[candidate.action_index] = _add_selection_reason(candidate, reason)
            order.append(candidate.action_index)
            return True
        had_reason = reason in existing.selection_reasons
        selected[candidate.action_index] = _add_selection_reason(existing, reason)
        return not had_reason

    def add_ranked(candidates: tuple[EarlySearchCandidate, ...], reason: str, limit: int) -> None:
        added = 0
        for candidate in candidates:
            if added >= limit:
                break
            if add(candidate, reason):
                added += 1

    add_ranked(_baseline_candidates(state, candidate_by_action), "baseline-keep", config.draw_baseline_keep)
    add_ranked(_rank_by_draw_score(state, ranked_by_pattern, _draw_safety_score), "safety-keep", config.draw_safety_keep)
    add_ranked(_rank_by_draw_score(state, ranked_by_pattern, _bottom_strengthening_score), "bottom-strengthening-keep", 1)
    add_ranked(_rank_by_draw_score(state, ranked_by_pattern, _middle_stabilizing_score), "middle-stabilizing-keep", 1)
    add_ranked(_rank_by_draw_score(state, ranked_by_pattern, _top_flexibility_score), "top-flexibility-keep", 1)
    add_ranked(_rank_by_draw_score(state, ranked_by_pattern, _capacity_flexibility_score), "capacity-flexibility-keep", 1)
    add_ranked(ranked_by_pattern, "pattern-prior", config.beam_size)

    del order
    return tuple(selected[index] for index in sorted(selected))


def _add_selection_reason(candidate: EarlySearchCandidate, reason: str) -> EarlySearchCandidate:
    if reason in candidate.selection_reasons:
        return candidate
    return EarlySearchCandidate(
        action_index=candidate.action_index,
        action=candidate.action,
        pattern_score=candidate.pattern_score,
        reasons=candidate.reasons,
        candidate_rank=candidate.candidate_rank,
        selection_reasons=candidate.selection_reasons + (reason,),
    )


def _baseline_candidates(
    state: GameState,
    candidate_by_action: dict[GameAction, EarlySearchCandidate],
) -> tuple[EarlySearchCandidate, ...]:
    ranked_actions = HeuristicRolloutPolicy().rank_actions(state)
    return tuple(
        candidate_by_action[scored.action]
        for scored in ranked_actions
        if scored.action in candidate_by_action
    )


def _rank_by_draw_score(
    state: GameState,
    candidates: tuple[EarlySearchCandidate, ...],
    score_fn,
) -> tuple[EarlySearchCandidate, ...]:
    scored = tuple(
        (score_fn(state, candidate.action), candidate)
        for candidate in candidates
    )
    return tuple(
        candidate
        for score, candidate in sorted(
            scored,
            key=lambda item: (-item[0], item[1].action_index),
        )
        if score > 0.0
    )


def _draw_safety_score(state: GameState, action: GameAction) -> float:
    next_state = apply_action(state, action)
    board = get_player(next_state, state.acting_player).board
    root_risk = score_root_action(state, action, config=RootRiskConfig.default())
    safe_score, _ = _safe_structure_score(board)
    return (
        20.0
        + root_risk.contribution
        + 1.4 * safe_score
        + _top_flexibility_score(state, action)
        + _capacity_flexibility_score(state, action)
    )


def _bottom_strengthening_score(state: GameState, action: GameAction) -> float:
    before = get_player(state, state.acting_player).board
    after = get_player(apply_action(state, action), state.acting_player).board
    placed_bottom = max(0, len(after.bottom) - len(before.bottom))
    if placed_bottom == 0:
        return 0.0
    return 10.0 * placed_bottom + max(0.0, _support_rank(after.bottom, RowName.BOTTOM) - _support_rank(before.bottom, RowName.BOTTOM))


def _middle_stabilizing_score(state: GameState, action: GameAction) -> float:
    before = get_player(state, state.acting_player).board
    after = get_player(apply_action(state, action), state.acting_player).board
    placed_middle = max(0, len(after.middle) - len(before.middle))
    if placed_middle == 0:
        return 0.0
    bottom_support = _support_rank(after.bottom, RowName.BOTTOM)
    middle_support = _support_rank(after.middle, RowName.MIDDLE)
    if middle_support > bottom_support + 8.0:
        return 0.0
    return 7.0 * placed_middle + max(0.0, bottom_support - middle_support)


def _top_flexibility_score(state: GameState, action: GameAction) -> float:
    before = get_player(state, state.acting_player).board
    after = get_player(apply_action(state, action), state.acting_player).board
    cards_after = board_card_count(after)
    if len(after.top) == len(before.top):
        return 8.0
    if len(after.top) < state.config.top_row_capacity:
        return 4.0
    if cards_after <= 9:
        return 0.0
    return 2.0


def _capacity_flexibility_score(state: GameState, action: GameAction) -> float:
    after = get_player(apply_action(state, action), state.acting_player).board
    top_open = state.config.top_row_capacity - len(after.top)
    middle_open = state.config.middle_row_capacity - len(after.middle)
    bottom_open = state.config.bottom_row_capacity - len(after.bottom)
    if min(top_open, middle_open, bottom_open) < 0:
        return 0.0
    return 0.8 * top_open + 0.5 * middle_open + 0.5 * bottom_open


def detect_card_patterns(cards: tuple[Card, ...]) -> PatternAssessment:
    """Detect made and drawing structures in a card group."""

    if not cards:
        return PatternAssessment(score=0.0, reasons=())

    score = 0.0
    reasons: list[str] = []
    rank_counts = Counter(int(card.rank) for card in cards)
    count_values = sorted(rank_counts.values(), reverse=True)

    if count_values and count_values[0] >= 4:
        score += 72.0
        reasons.append("quads")
    if _has_full_house_like(count_values):
        score += 62.0
        reasons.append("full-house-like")
    if count_values and count_values[0] >= 3:
        trip_rank = max(rank for rank, count in rank_counts.items() if count >= 3)
        score += 38.0 + _rank_bonus(trip_rank)
        reasons.append("trips")
    pair_ranks = tuple(rank for rank, count in rank_counts.items() if count >= 2)
    if len(pair_ranks) >= 2:
        score += 30.0 + _rank_bonus(max(pair_ranks))
        reasons.append("two-pair")
    elif len(pair_ranks) == 1:
        score += 14.0 + _rank_bonus(pair_ranks[0])
        reasons.append("pair")

    flush_score, flush_reasons = _flush_patterns(cards)
    score += flush_score
    reasons.extend(flush_reasons)

    straight_score, straight_reasons = _straight_patterns(cards)
    score += straight_score
    reasons.extend(straight_reasons)

    cluster_score, cluster_reason = _rank_cluster_pattern(rank_counts)
    if cluster_reason:
        score += cluster_score
        reasons.append(cluster_reason)

    high_cards = sum(1 for card in cards if card.rank >= Rank.QUEEN)
    if high_cards:
        score += 3.0 * high_cards
        reasons.append(f"high-cards-{high_cards}")

    return PatternAssessment(score=score, reasons=tuple(dict.fromkeys(reasons)))


def _score_action(state: GameState, action_index: int, action: GameAction) -> EarlySearchCandidate:
    player_before = get_player(state, state.acting_player)
    next_state = apply_action(state, action)
    player_after = get_player(next_state, state.acting_player)
    board = player_after.board

    score = 0.0
    reasons: list[str] = []
    for row in ROW_ORDER:
        row_score, row_reasons = _row_pattern_score(board, row)
        score += row_score
        reasons.extend(row_reasons)

    score += _development_score(player_before.board, board)
    safety_score, safety_reasons = _safe_structure_score(board)
    score += safety_score
    reasons.extend(safety_reasons)

    if not reasons:
        reasons.append("neutral")
    return EarlySearchCandidate(
        action_index=action_index,
        action=action,
        pattern_score=score,
        reasons=tuple(dict.fromkeys(reasons)),
        candidate_rank=0,
    )


def _row_pattern_score(board: Board, row: RowName) -> tuple[float, list[str]]:
    cards = row_cards(board, row)
    assessment = detect_card_patterns(cards)
    if not cards:
        return (0.0, [])

    if row is RowName.BOTTOM:
        weight = 1.4
    elif row is RowName.MIDDLE:
        weight = 0.92
    else:
        weight = 0.38

    score = assessment.score * weight
    reasons = [f"{row.value}:{reason}" for reason in assessment.reasons if _reason_is_structural(reason)]

    if row is RowName.TOP:
        top_score, top_reasons = _top_potential(cards)
        score += top_score
        reasons.extend(top_reasons)
    if row is RowName.BOTTOM and len(cards) >= 3:
        score += 2.0 * len(cards)
        reasons.append("bottom:strength-foundation")
    return (score, reasons)


def _development_score(before: Board, after: Board) -> float:
    before_count = board_card_count(before)
    after_count = board_card_count(after)
    if after_count <= before_count:
        return 0.0

    score = 0.0
    if len(after.bottom) >= len(after.middle):
        score += 8.0
    if len(after.bottom) >= 3 and after_count <= 7:
        score += 5.0
    if len(after.top) == 3 and after_count <= 7:
        score -= 8.0
    return score


def _safe_structure_score(board: Board) -> tuple[float, list[str]]:
    reasons: list[str] = []
    score = 0.0
    bottom_support = _support_rank(board.bottom, RowName.BOTTOM)
    middle_support = _support_rank(board.middle, RowName.MIDDLE)
    top_support = _support_rank(board.top, RowName.TOP)

    if bottom_support >= middle_support:
        score += 5.0
        reasons.append("safe:bottom-support")
    else:
        gap = middle_support - bottom_support
        score -= 3.0 + 0.25 * gap
        reasons.append("risk:middle-over-bottom")

    if middle_support >= top_support:
        score += 3.0
        reasons.append("safe:middle-support")
    else:
        gap = top_support - middle_support
        score -= 2.0 + 0.2 * gap
        reasons.append("risk:top-over-middle")
    return score, reasons


def _top_potential(cards: tuple[Card, ...]) -> tuple[float, list[str]]:
    if not cards:
        return (0.0, [])

    reasons: list[str] = []
    score = 0.0
    counts = Counter(int(card.rank) for card in cards)
    pairs = [rank for rank, count in counts.items() if count == 2]
    trips = [rank for rank, count in counts.items() if count == 3]
    if trips:
        score += 26.0 + _rank_bonus(max(trips))
        reasons.append("top:trips")
    if pairs:
        pair_rank = max(pairs)
        score += 12.0 + _rank_bonus(pair_rank)
        reasons.append("top:pair")
        if pair_rank >= int(Rank.QUEEN):
            score += 18.0
            reasons.append("top:qq-plus")
    high_cards = sum(1 for card in cards if card.rank >= Rank.QUEEN)
    if high_cards and len(cards) < 3:
        score += 5.0 * high_cards
        reasons.append("top:high-card-flex")
    return score, reasons


def _support_rank(cards: tuple[Card, ...], row: RowName) -> float:
    if not cards:
        return 0.0
    if row is RowName.TOP and len(cards) == 3:
        value = evaluate_top_row(cards)
        return float(20 * int(value.category) + sum(value.tiebreak) / 20)
    if len(cards) == 5:
        value = evaluate_five_card_row(cards)
        return float(20 * int(value.category) + sum(value.tiebreak) / 20)
    assessment = detect_card_patterns(cards)
    return assessment.score / 5.0 + max(int(card.rank) for card in cards) / 4.0


def _has_full_house_like(count_values: list[int]) -> bool:
    if not count_values or count_values[0] < 3:
        return False
    remaining = count_values[1:]
    return any(count >= 2 for count in remaining) or count_values[0] >= 5


def _flush_patterns(cards: tuple[Card, ...]) -> tuple[float, list[str]]:
    suit_counts = Counter(card.suit for card in cards)
    max_count = max(suit_counts.values())
    if max_count >= 5:
        return (44.0, ["flush-made"])
    if max_count == 4:
        return (24.0, ["flush-draw-4"])
    if max_count == 3:
        return (10.0, ["flush-draw-3"])
    return (0.0, [])


def _straight_patterns(cards: tuple[Card, ...]) -> tuple[float, list[str]]:
    ranks = {int(card.rank) for card in cards}
    if int(Rank.ACE) in ranks:
        ranks.add(1)
    longest = 1
    for start in range(1, 11):
        count = sum(1 for rank in range(start, start + 5) if rank in ranks)
        longest = max(longest, count)
    if longest >= 5:
        return (40.0, ["straight-made"])
    if longest == 4:
        return (22.0, ["straight-draw-4"])
    if longest == 3:
        return (8.0, ["straight-draw-3"])
    return (0.0, [])


def _rank_cluster_pattern(rank_counts: Counter[int]) -> tuple[float, str | None]:
    ranks = sorted(rank_counts)
    best = 1
    for left, rank in enumerate(ranks):
        right = left
        while right + 1 < len(ranks) and ranks[right + 1] - rank <= 4:
            right += 1
        best = max(best, right - left + 1)
    if best >= 4:
        return (8.0, "rank-cluster-4")
    if best == 3:
        return (3.0, "rank-cluster-3")
    return (0.0, None)


def _rank_bonus(rank: int) -> float:
    return max(0.0, (rank - 7) * 1.5)


def _reason_is_structural(reason: str) -> bool:
    return (
        reason in {"quads", "full-house-like", "trips", "two-pair", "pair", "flush-made", "straight-made"}
        or reason.startswith("flush-draw")
        or reason.startswith("straight-draw")
        or reason.startswith("rank-cluster")
    )


__all__ = [
    "EarlySearchCandidate",
    "EarlySearchCandidateSet",
    "EarlySearchConfig",
    "PatternAssessment",
    "detect_card_patterns",
    "select_early_search_candidates",
]
