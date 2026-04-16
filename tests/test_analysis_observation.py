from __future__ import annotations

import unittest

from ofc.board import Board, visible_cards
from ofc.cards import format_card, parse_card
from ofc.config import DEFAULT_CONFIG
from ofc.deck import DeckState
from ofc.state import GameState, HandPhase, PlayerId, PlayerState
from ofc_analysis.observation import project_observation
from tests.helpers import cards, make_board, remaining_deck_tokens


def _make_concealed_state() -> GameState:
    concealed_board = make_board(
        top="Ah Ad Ac",
        middle="Kh Kd Ks Qc Qd",
        bottom="2h 3h 4h 5h 6h",
    )
    visible_board = make_board(
        top="Js Td",
        middle="9c 8d",
        bottom="7s",
    )
    left_player = PlayerState(
        player_id=PlayerId.PLAYER_0,
        board=Board(),
        hidden_discards=cards("7c"),
        current_private_draw=(),
        fantasyland_active=True,
        concealed_fantasyland_board=concealed_board,
        concealed_fantasyland_discard=parse_card("7c"),
        initial_placement_done=False,
        normal_draws_taken=0,
        fantasyland_set_done=True,
    )
    right_player = PlayerState(
        player_id=PlayerId.PLAYER_1,
        board=visible_board,
        hidden_discards=cards("Ts"),
        current_private_draw=cards("As Kc Qh"),
        fantasyland_active=False,
        initial_placement_done=True,
        normal_draws_taken=1,
        fantasyland_set_done=False,
    )
    used_cards = (
        visible_cards(concealed_board)
        + left_player.hidden_discards
        + visible_cards(visible_board)
        + right_player.hidden_discards
        + right_player.current_private_draw
    )
    deck = DeckState(undealt_cards=cards(remaining_deck_tokens(format_card(card) for card in used_cards)))
    return GameState(
        config=DEFAULT_CONFIG,
        hand_number=3,
        button=PlayerId.PLAYER_0,
        acting_player=PlayerId.PLAYER_1,
        phase=HandPhase.DRAW,
        deck=deck,
        players=(left_player, right_player),
        is_continuation_hand=True,
        next_hand_fantasyland=(True, False),
    )


class AnalysisObservationTest(unittest.TestCase):
    def test_project_observation_preserves_observer_private_fantasyland_information(self) -> None:
        state = _make_concealed_state()

        observation = project_observation(state, PlayerId.PLAYER_0)

        self.assertEqual(PlayerId.PLAYER_0, observation.observer)
        self.assertEqual((PlayerId.PLAYER_0, PlayerId.PLAYER_1), observation.public_player_ids)
        self.assertEqual(Board(), observation.public_boards[0])
        self.assertEqual(cards("7c"), observation.own_hidden_discards)
        self.assertEqual(parse_card("7c"), observation.own_concealed_fantasyland_discard)
        self.assertIsNotNone(observation.own_concealed_fantasyland_board)
        self.assertEqual(1, observation.opponent_hidden_discard_count)
        self.assertEqual(33, observation.unseen_card_count)

    def test_project_observation_redacts_opponent_private_information(self) -> None:
        state = _make_concealed_state()

        observation = project_observation(state, PlayerId.PLAYER_1)

        self.assertEqual(cards("As Kc Qh"), observation.own_private_draw)
        self.assertEqual(cards("Ts"), observation.own_hidden_discards)
        self.assertEqual(Board(), observation.public_boards[0])
        self.assertIsNone(observation.own_concealed_fantasyland_board)
        self.assertIsNone(observation.own_concealed_fantasyland_discard)
        self.assertEqual(1, observation.opponent_hidden_discard_count)
        self.assertEqual(43, observation.unseen_card_count)


if __name__ == "__main__":
    unittest.main()
