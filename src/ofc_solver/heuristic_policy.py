"""Greedy rollout policy with interpretable OFC board-shape heuristics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import random

from ofc.actions import GameAction, Placement, PlaceDrawAction, SetFantasylandHandAction
from ofc.board import Board, RowName, board_card_count, board_full, row_capacity, visible_cards
from ofc.cards import Card, Rank
from ofc.config import VariantConfig
from ofc.engine import showdown
from ofc.evaluator import (
    HandCategory,
    compare_row_values,
    evaluate_five_card_row,
    evaluate_top_row,
)
from ofc.fantasyland import qualifies_for_fantasyland, qualifies_to_stay_in_fantasyland
from ofc.scoring import is_foul, royalties_for_board
from ofc.state import GameState, HandPhase, PlayerId, get_player
from ofc.transitions import apply_action, validate_action, legal_actions


@dataclass(frozen=True)
class ActionScore:
    """Debuggable score assigned to a candidate rollout action."""

    action: GameAction
    score: float
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class PolicyDecisionDiagnostics:
    """Debug metadata for one rollout-policy decision."""

    selected_score: float
    selected_reasons: tuple[str, ...] = ()
    used_exact_late_search: bool = False
    exact_late_search_node_count: int = 0


@dataclass(frozen=True)
class HeuristicRolloutPolicy:
    """Deterministic greedy rollout policy with seeded tie-breaking."""

    fantasyland_bottom_options: int = 10
    exact_search_max_decisions: int = 3
    exact_search_node_limit: int = 100

    def choose_action(self, state: GameState, *, rng: random.Random) -> GameAction:
        """Return the highest-scoring legal rollout action."""

        action, _ = self.choose_action_with_diagnostics(state, rng=rng)
        return action

    def choose_action_with_diagnostics(
        self,
        state: GameState,
        *,
        rng: random.Random,
    ) -> tuple[GameAction, PolicyDecisionDiagnostics]:
        """Return the selected action plus decision diagnostics."""

        scored_actions, exact_node_count = self._rank_actions_with_context(state)
        if not scored_actions:
            raise ValueError(f"No rollout actions are available during {state.phase.value}")

        best_score = max(scored.score for scored in scored_actions)
        tied_best = tuple(scored for scored in scored_actions if scored.score == best_score)
        selected = tied_best[rng.randrange(len(tied_best))]
        used_exact_search = "exact-late-street" in selected.reasons
        return selected.action, PolicyDecisionDiagnostics(
            selected_score=selected.score,
            selected_reasons=selected.reasons,
            used_exact_late_search=used_exact_search,
            exact_late_search_node_count=exact_node_count if used_exact_search else 0,
        )

    def rank_actions(self, state: GameState) -> tuple[ActionScore, ...]:
        """Return candidate actions sorted by heuristic score descending."""

        scored_actions, _ = self._rank_actions_with_context(state)
        return scored_actions

    def _rank_actions_with_context(self, state: GameState) -> tuple[tuple[ActionScore, ...], int]:
        exact_node_count = self._exact_late_search_node_count(state)
        if exact_node_count is not None:
            return self._rank_exact_late_actions(state), exact_node_count

        if state.phase == HandPhase.FANTASYLAND_SET:
            actions = self._fantasyland_candidate_actions(state)
        else:
            actions = tuple(legal_actions(state))
        scored = tuple(self._score_action(state, action) for action in actions)
        return tuple(sorted(scored, key=lambda item: -item.score)), 0

    def _exact_late_search_node_count(self, state: GameState) -> int | None:
        remaining_decisions = _remaining_normal_draw_decisions(state)
        if remaining_decisions is None:
            return None
        if remaining_decisions > self.exact_search_max_decisions:
            return None
        node_count = _exact_tree_node_count(state, self.exact_search_node_limit)
        if node_count > self.exact_search_node_limit:
            return None
        return node_count

    def _rank_exact_late_actions(self, state: GameState) -> tuple[ActionScore, ...]:
        perspective = state.acting_player
        scored = tuple(
            ActionScore(
                action=action,
                score=_exact_late_value(apply_action(state, action), perspective),
                reasons=("exact-late-street",),
            )
            for action in legal_actions(state)
        )
        return tuple(sorted(scored, key=lambda item: -item.score))

    def _score_action(self, state: GameState, action: GameAction) -> ActionScore:
        acting_player = state.acting_player
        next_state = apply_action(state, action)
        if next_state.phase in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
            _, result = showdown(next_state)
            return ActionScore(
                action=action,
                score=_terminal_value_for_player(result, acting_player),
                reasons=("terminal-value",),
            )

        player_after = get_player(next_state, acting_player)
        board = (
            player_after.concealed_fantasyland_board
            if isinstance(action, SetFantasylandHandAction)
            else player_after.board
        )
        if board is None:
            raise ValueError("Fantasyland set action did not produce a concealed board")

        score, reasons = _score_board(
            board,
            config=state.config,
            fantasyland_active=player_after.fantasyland_active,
        )
        if isinstance(action, (PlaceDrawAction, SetFantasylandHandAction)):
            discard_score, discard_reasons = _score_discard(action.discard, board)
            score += discard_score
            reasons.extend(discard_reasons)
        return ActionScore(action=action, score=score, reasons=tuple(reasons))

    def _fantasyland_candidate_actions(self, state: GameState) -> tuple[GameAction, ...]:
        if state.phase != HandPhase.FANTASYLAND_SET:
            raise ValueError("Fantasyland candidates require FANTASYLAND_SET phase")

        player = get_player(state, state.acting_player)
        cards = tuple(player.current_private_draw)
        if len(cards) != state.config.fantasyland_deal_count:
            raise ValueError("Fantasyland heuristic requires a 14-card private draw")

        actions: list[SetFantasylandHandAction] = []
        seen: set[tuple[tuple[Card, ...], tuple[Card, ...], tuple[Card, ...], Card]] = set()

        for top_cards in _fantasyland_top_candidates(cards):
            remaining_after_top = _without(cards, top_cards)
            for bottom_cards in _ranked_five_card_options(remaining_after_top)[: self.fantasyland_bottom_options]:
                remaining_after_bottom = _without(remaining_after_top, bottom_cards)
                for middle_cards in _ranked_five_card_options(remaining_after_bottom):
                    discard_cards = _without(remaining_after_bottom, middle_cards)
                    if len(discard_cards) != 1:
                        continue
                    key = (
                        _canonical_cards(top_cards),
                        _canonical_cards(middle_cards),
                        _canonical_cards(bottom_cards),
                        discard_cards[0],
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    action = _set_fantasyland_action(
                        player.player_id.value,
                        top_cards=top_cards,
                        middle_cards=middle_cards,
                        bottom_cards=bottom_cards,
                        discard=discard_cards[0],
                    )
                    validate_action(state, action)
                    actions.append(action)

        if actions:
            return tuple(actions)

        # Defensive fallback: this should only be reachable for malformed states,
        # but keeping it deterministic makes the policy easier to debug.
        sorted_cards = tuple(sorted(cards, key=_card_sort_key, reverse=True))
        return (
            _set_fantasyland_action(
                player.player_id.value,
                top_cards=sorted_cards[10:13],
                middle_cards=sorted_cards[5:10],
                bottom_cards=sorted_cards[:5],
                discard=sorted_cards[13],
            ),
        )


def _score_board(board: Board, *, config: VariantConfig, fantasyland_active: bool) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []

    if board_full(board, config):
        if is_foul(board):
            score -= 10000.0
            reasons.append("completed-foul")
        else:
            royalties = royalties_for_board(board, config)
            if royalties:
                score += 40.0 * royalties
                reasons.append(f"royalties:{royalties}")
            if fantasyland_active and qualifies_to_stay_in_fantasyland(board):
                score += 500.0
                reasons.append("fantasyland-stay")
            elif not fantasyland_active and qualifies_for_fantasyland(board):
                score += 500.0
                reasons.append("fantasyland-entry")

    if len(board.middle) == config.middle_row_capacity and len(board.top) == config.top_row_capacity:
        if _compare_cross_rows_cached(board.middle, board.top) < 0:
            score -= 4000.0
            reasons.append("middle-below-top")

    if len(board.bottom) == config.bottom_row_capacity and len(board.middle) == config.middle_row_capacity:
        if _compare_cross_rows_cached(board.bottom, board.middle) < 0:
            score -= 4000.0
            reasons.append("bottom-below-middle")

    cards_on_board = board_card_count(board)
    score += 1.25 * _five_row_score(board.bottom, RowName.BOTTOM, config)
    score += 1.00 * _five_row_score(board.middle, RowName.MIDDLE, config)
    score += _top_row_score(board.top, board=board, config=config)
    score += _open_capacity_bonus(board, config=config, cards_on_board=cards_on_board)
    score -= _shape_penalty(board, config=config)
    if not board_full(board, config):
        survivability_penalty, survivability_reasons = _survivability_penalty(board, config=config)
        score -= survivability_penalty
        reasons.extend(survivability_reasons)
    return score, reasons


def _terminal_value_for_player(result, player_id: PlayerId) -> float:
    if PlayerId(result.left.player_id) == player_id:
        return float(result.left.total_points)
    if PlayerId(result.right.player_id) == player_id:
        return float(result.right.total_points)
    raise ValueError(f"Terminal result does not contain player {player_id.value}")


def _remaining_normal_draw_decisions(state: GameState) -> int | None:
    if state.phase != HandPhase.DRAW:
        return None

    total = 0
    for player in state.players:
        if player.current_private_draw and player.player_id != state.acting_player:
            return None
        if player.fantasyland_active:
            if not player.fantasyland_set_done:
                return None
            continue
        if not player.initial_placement_done:
            return None
        remaining = state.config.normal_draw_turns_per_player - player.normal_draws_taken
        if remaining < 0:
            return None
        total += remaining
    return total


def _exact_tree_node_count(state: GameState, limit: int) -> int:
    if state.phase in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        return 1
    if state.phase != HandPhase.DRAW:
        return limit + 1

    total = 1
    actions = tuple(legal_actions(state))
    if not actions:
        return limit + 1
    for action in actions:
        total += _exact_tree_node_count(apply_action(state, action), limit - total)
        if total > limit:
            return total
    return total


def _exact_late_value(state: GameState, perspective: PlayerId) -> float:
    if state.phase in {HandPhase.SHOWDOWN, HandPhase.TERMINAL}:
        _, result = showdown(state)
        return _terminal_value_for_player(result, perspective)
    if state.phase != HandPhase.DRAW:
        raise ValueError(f"Exact late-street search cannot evaluate {state.phase.value}")

    child_values = tuple(_exact_late_value(apply_action(state, action), perspective) for action in legal_actions(state))
    if not child_values:
        raise ValueError("Exact late-street search found no legal actions")
    if state.acting_player == perspective:
        return max(child_values)
    return min(child_values)


def _five_row_score(cards: tuple[Card, ...], row: RowName, config: VariantConfig) -> float:
    if not cards:
        return 0.0
    if len(cards) == row_capacity(row, config):
        value = _cached_five_value(cards)
        return 65.0 * int(value.category) + 1.6 * sum(value.tiebreak)

    ranks = [int(card.rank) for card in cards]
    counts = Counter(ranks)
    score = 1.2 * sum(ranks)

    pair_ranks = [rank for rank, count in counts.items() if count == 2]
    trip_ranks = [rank for rank, count in counts.items() if count == 3]
    quad_ranks = [rank for rank, count in counts.items() if count == 4]
    score += sum(18.0 + 1.2 * rank for rank in pair_ranks)
    score += sum(58.0 + 1.8 * rank for rank in trip_ranks)
    score += sum(105.0 + 2.0 * rank for rank in quad_ranks)
    if len(pair_ranks) >= 2:
        score += 28.0

    max_suited = max(Counter(card.suit for card in cards).values(), default=0)
    if max_suited >= 3:
        score += 8.0 * (max_suited - 2)
    if max_suited == 4:
        score += 18.0

    straight_density = _straight_density_score(ranks)
    score += straight_density
    if row is RowName.BOTTOM:
        score += 3.0 * len(cards)
    return score


def _top_row_score(cards: tuple[Card, ...], *, board: Board, config: VariantConfig) -> float:
    if not cards:
        return 0.0
    ranks = [int(card.rank) for card in cards]
    counts = Counter(ranks)

    if len(cards) == config.top_row_capacity:
        value = _cached_top_value(cards)
        score = 42.0 * int(value.category) + 2.0 * sum(value.tiebreak)
        if value.category == HandCategory.ONE_PAIR:
            pair_rank = value.tiebreak[0]
            score += 155.0 if pair_rank >= int(Rank.QUEEN) else 10.0 + pair_rank
        elif value.category == HandCategory.THREE_OF_A_KIND:
            score += 230.0
        return score

    score = 0.35 * sum(ranks)
    pair_ranks = [rank for rank, count in counts.items() if count == 2]
    if pair_ranks:
        pair_rank = max(pair_ranks)
        score += 50.0 if pair_rank >= int(Rank.QUEEN) else 12.0 + pair_rank
    if any(count == 3 for count in counts.values()):
        score += 90.0
    return score


def _open_capacity_bonus(board: Board, *, config: VariantConfig, cards_on_board: int) -> float:
    if cards_on_board >= 10:
        scale = 0.4
    elif cards_on_board >= 7:
        scale = 0.9
    else:
        scale = 1.6
    return scale * (
        row_capacity(RowName.BOTTOM, config)
        - len(board.bottom)
        + row_capacity(RowName.MIDDLE, config)
        - len(board.middle)
        + 0.5 * (row_capacity(RowName.TOP, config) - len(board.top))
    )


def _shape_penalty(board: Board, *, config: VariantConfig) -> float:
    penalty = 0.0
    top_strength = _top_threat_rank(board.top, config=config)
    if top_strength:
        middle_support = _support_rank(board.middle)
        bottom_support = _support_rank(board.bottom)
        if middle_support < top_strength:
            penalty += 24.0 + 5.0 * (top_strength - middle_support)
        if bottom_support + 1 < middle_support:
            penalty += 12.0 + 4.0 * (middle_support - bottom_support)

    if board.top and board_card_count(board) <= 7:
        middle_support = _support_rank(board.middle)
        bottom_support = _support_rank(board.bottom)
        if middle_support < 10 and bottom_support < 10:
            penalty += sum(max(0, int(card.rank) - 10) for card in board.top) * 5.0

    if len(board.bottom) >= 4 and len(board.middle) >= 4:
        bottom_score = _five_row_score(board.bottom, RowName.BOTTOM, config)
        middle_score = _five_row_score(board.middle, RowName.MIDDLE, config)
        if bottom_score + 35.0 < middle_score:
            penalty += 45.0 + 0.25 * (middle_score - bottom_score)
    return penalty


def _survivability_penalty(board: Board, *, config: VariantConfig) -> tuple[float, list[str]]:
    """Penalize incomplete shapes that are likely to paint lower rows into a corner."""

    penalty = 0.0
    reasons: list[str] = []
    middle_support = _support_rank(board.middle)
    bottom_support = _support_rank(board.bottom)
    top_counts = Counter(int(card.rank) for card in board.top)

    top_pairs = [rank for rank, count in top_counts.items() if count == 2]
    if top_pairs:
        pair_rank = max(top_pairs)
        if pair_rank >= int(Rank.QUEEN) and (middle_support < pair_rank or bottom_support < pair_rank):
            support_gap = pair_rank - min(middle_support, bottom_support)
            penalty += 220.0 + 12.0 * max(0, support_gap)
            if len(board.top) == config.top_row_capacity:
                penalty += 55.0
            reasons.append("unsupported-top-pair")

    top_trips = [rank for rank, count in top_counts.items() if count == 3]
    if top_trips:
        trip_rank = max(top_trips)
        needed_support = trip_rank + 4
        if middle_support < needed_support or bottom_support < needed_support:
            support_gap = needed_support - min(middle_support, bottom_support)
            penalty += 450.0 + 10.0 * max(0, support_gap)
            reasons.append("unsupported-top-trips")

    if len(board.top) == config.top_row_capacity and _top_threat_rank(board.top, config=config):
        if middle_support < _top_threat_rank(board.top, config=config) or bottom_support + 1 < middle_support:
            penalty += 70.0
            reasons.append("completed-top-pressure")

    if board.middle and board.bottom and len(board.bottom) < config.bottom_row_capacity:
        support_gap = middle_support - bottom_support
        if support_gap >= 4:
            penalty += 120.0 + 9.0 * support_gap
            reasons.append("middle-over-bottom-pressure")

    if board_card_count(board) <= 10 and len(board.bottom) < len(board.middle):
        support_gap = max(0, middle_support - bottom_support)
        penalty += 22.0 * (len(board.middle) - len(board.bottom)) + 5.0 * support_gap
        reasons.append("bottom-underbuilt")

    return penalty, reasons


def _top_threat_rank(cards: tuple[Card, ...], *, config: VariantConfig) -> int:
    if not cards:
        return 0
    counts = Counter(int(card.rank) for card in cards)
    trips = [rank for rank, count in counts.items() if count == 3]
    if trips:
        return max(trips) + 7
    pairs = [rank for rank, count in counts.items() if count == 2]
    if pairs:
        return max(pairs)
    if len(cards) == config.top_row_capacity:
        return 0
    high_cards = [rank for rank in counts if rank >= int(Rank.QUEEN)]
    return max(high_cards, default=0) - 2


def _support_rank(cards: tuple[Card, ...]) -> int:
    if not cards:
        return 0
    counts = Counter(int(card.rank) for card in cards)
    if any(count >= 3 for count in counts.values()):
        return max(rank for rank, count in counts.items() if count >= 3) + 4
    if any(count == 2 for count in counts.values()):
        return max(rank for rank, count in counts.items() if count == 2)
    return max(counts) - 3


def _score_discard(discard: Card, board: Board) -> tuple[float, list[str]]:
    board_cards = visible_cards(board)
    rank = int(discard.rank)
    penalty = 0.35 * rank
    reasons: list[str] = []

    same_rank_on_board = sum(1 for card in board_cards if card.rank == discard.rank)
    if same_rank_on_board == 1:
        penalty += 7.0 + 0.6 * rank
        reasons.append("discard-breaks-pair")
    elif same_rank_on_board >= 2:
        penalty += 24.0 + 1.0 * rank
        reasons.append("discard-breaks-trips")

    for row_cards in (board.middle, board.bottom):
        suited_count = sum(1 for card in row_cards if card.suit == discard.suit)
        if suited_count >= 4:
            penalty += 18.0
            reasons.append("discard-breaks-flush")
        elif suited_count == 3:
            penalty += 6.0

        ranks = [int(card.rank) for card in row_cards]
        if _extends_straight_window(rank, ranks):
            penalty += 8.0
            reasons.append("discard-breaks-straight")

    if rank <= 6 and same_rank_on_board == 0 and not any(
        _extends_straight_window(rank, [int(card.rank) for card in row_cards])
        for row_cards in (board.middle, board.bottom)
    ):
        penalty -= 5.0
        reasons.append("low-disconnected-discard")

    return -penalty, reasons


def _straight_density_score(ranks: list[int]) -> float:
    unique = set(ranks)
    if 14 in unique:
        unique.add(1)
    best = 0
    for start in range(1, 11):
        best = max(best, sum(1 for rank in range(start, start + 5) if rank in unique))
    if best <= 2:
        return 0.0
    return 4.0 * best


def _extends_straight_window(rank: int, row_ranks: list[int]) -> bool:
    ranks = set(row_ranks)
    if not ranks:
        return False
    test_ranks = set(ranks)
    test_ranks.add(rank)
    if 14 in test_ranks:
        test_ranks.add(1)
    for start in range(1, 11):
        window = set(range(start, start + 5))
        if rank in window and len(test_ranks & window) >= 4:
            return True
    return False


def _fantasyland_top_candidates(cards: tuple[Card, ...]) -> tuple[tuple[Card, ...], ...]:
    return _fantasyland_top_candidates_cached(_canonical_cards(cards))


@lru_cache(maxsize=8192)
def _fantasyland_top_candidates_cached(cards: tuple[Card, ...]) -> tuple[tuple[Card, ...], ...]:
    all_top_sets = tuple(combinations(cards, 3))
    selected: list[tuple[Card, ...]] = []

    weak_top_sets = sorted(all_top_sets, key=_top_strength_sort_key)[:12]
    strong_top_sets = sorted(all_top_sets, key=_top_strength_sort_key, reverse=True)[:18]
    selected.extend(weak_top_sets)
    selected.extend(strong_top_sets)

    for candidate in all_top_sets:
        counts = Counter(int(card.rank) for card in candidate)
        if 3 in counts.values():
            selected.append(candidate)
        elif any(count == 2 and rank >= int(Rank.QUEEN) for rank, count in counts.items()):
            selected.append(candidate)

    return _dedupe_card_groups(selected)


def _ranked_five_card_options(cards: tuple[Card, ...]) -> tuple[tuple[Card, ...], ...]:
    return _ranked_five_card_options_cached(_canonical_cards(cards))


@lru_cache(maxsize=65536)
def _ranked_five_card_options_cached(cards: tuple[Card, ...]) -> tuple[tuple[Card, ...], ...]:
    return tuple(sorted(combinations(cards, 5), key=_five_strength_sort_key, reverse=True))


def _top_strength_sort_key(cards: tuple[Card, ...]) -> tuple[int, tuple[int, ...]]:
    value = _cached_top_value(cards)
    return (int(value.category), value.tiebreak)


def _five_strength_sort_key(cards: tuple[Card, ...]) -> tuple[int, tuple[int, ...]]:
    value = _cached_five_value(cards)
    return (int(value.category), value.tiebreak)


def _compare_cross_rows_cached(left_cards: tuple[Card, ...], right_cards: tuple[Card, ...]) -> int:
    left_value = _cached_top_value(left_cards) if len(left_cards) == 3 else _cached_five_value(left_cards)
    right_value = _cached_top_value(right_cards) if len(right_cards) == 3 else _cached_five_value(right_cards)
    return compare_row_values(left_value, right_value)


def _cached_top_value(cards: tuple[Card, ...]):
    return _cached_top_value_canonical(_canonical_cards(cards))


@lru_cache(maxsize=65536)
def _cached_top_value_canonical(cards: tuple[Card, ...]):
    return evaluate_top_row(cards)


def _cached_five_value(cards: tuple[Card, ...]):
    return _cached_five_value_canonical(_canonical_cards(cards))


@lru_cache(maxsize=262144)
def _cached_five_value_canonical(cards: tuple[Card, ...]):
    return evaluate_five_card_row(cards)


def _set_fantasyland_action(
    player_id: str,
    *,
    top_cards: tuple[Card, ...],
    middle_cards: tuple[Card, ...],
    bottom_cards: tuple[Card, ...],
    discard: Card,
) -> SetFantasylandHandAction:
    placements = tuple(
        Placement(row=row, card=card)
        for row, row_cards in (
            (RowName.TOP, top_cards),
            (RowName.MIDDLE, middle_cards),
            (RowName.BOTTOM, bottom_cards),
        )
        for card in row_cards
    )
    return SetFantasylandHandAction(player_id=player_id, placements=placements, discard=discard)


def _dedupe_card_groups(groups) -> tuple[tuple[Card, ...], ...]:
    seen: set[tuple[Card, ...]] = set()
    deduped: list[tuple[Card, ...]] = []
    for group in groups:
        canonical = _canonical_cards(tuple(group))
        if canonical in seen:
            continue
        seen.add(canonical)
        deduped.append(tuple(group))
    return tuple(deduped)


def _without(cards: tuple[Card, ...], removed: tuple[Card, ...]) -> tuple[Card, ...]:
    removed_set = set(removed)
    return tuple(card for card in cards if card not in removed_set)


def _canonical_cards(cards: tuple[Card, ...]) -> tuple[Card, ...]:
    return tuple(sorted(cards, key=_card_sort_key))


def _card_sort_key(card: Card) -> tuple[int, str]:
    return (int(card.rank), card.suit.value)


__all__ = ["ActionScore", "HeuristicRolloutPolicy", "PolicyDecisionDiagnostics"]
