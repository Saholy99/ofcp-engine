# Heads-Up Pineapple Open-Face Chinese Poker (OFC) with Fantasyland

This document is the source-of-truth rules specification for the game engine.

## 1. Variant Summary

This engine implements **heads-up Pineapple OFC with Fantasyland** using:

- a standard **52-card deck**
- **2 players**
- **Top / Front** row of **3 cards**
- **Middle** row of **5 cards**
- **Bottom / Back** row of **5 cards**
- **Pineapple draw structure**:
  - each player starts with **5 cards**
  - on each later turn, the player receives **3 cards**
  - the player places **exactly 2** of those cards into their board
  - the player discards **exactly 1** of those cards **face down**
- **Fantasyland**
- **royalties / bonuses** per the tables below
- **standard 1–6 row/scoop scoring**

There are no betting rounds. The game is scored in points per hand.

---

## 2. Board Structure and Legality

Each player builds three rows:

- **Top / Front**: 3 cards
- **Middle**: 5 cards
- **Bottom / Back**: 5 cards

At the end of the hand, a player's board is **legal** if and only if:

- **Bottom >= Middle >= Top**

where comparison is by poker hand strength.

If this ordering condition is violated, the hand is a **foul**.

### 2.1 Row ranking domains

- **Top row** uses **3-card poker-style OFC ranking only**:
  - high card
  - pair
  - trips

  Straights and flushes do **not** count in the top row.

- **Middle** and **Bottom** rows use standard **5-card poker ranking**:
  - high card
  - pair
  - two pair
  - trips
  - straight
  - flush
  - full house
  - quads
  - straight flush
  - royal flush

### 2.2 Tie-breaking

Use standard poker tie-breaking rules.

- Suits never break ties.
- A-2-3-4-5 counts as a valid 5-high straight and straight flush.
- Exact row ties score **0** for that row.

### 2.3 Cross-row comparison for foul detection

Because foul detection compares rows of different sizes, use the following deterministic rule:

- compare by hand category first
- if categories differ, the stronger category is stronger
- if categories are the same, compare the made-hand ranks and kickers lexicographically using standard poker ordering
- when comparing a **3-card top row** against a **5-card middle row** with the same category, treat the 5-card row as having its full normal kicker structure and the 3-card row as having only its available cards

This is used only to determine whether:

- **Middle >= Top**
- **Bottom >= Middle**

Examples:
- Top = pair of Kings with Ace kicker
- Middle = pair of Kings with Queen-7-3 kickers

Then **Top > Middle**, so the hand is a foul.

---

## 3. Turn Order and Button

Use a dealer button.

- In a normal heads-up hand, the player **to the left of the button acts first**.
- In heads-up, this means the **non-button player acts first**.
- After a **normal non-Fantasyland continuation hand**, the button rotates.

### 3.1 Fantasyland continuation rule

If a Fantasyland hand occurs, the **button does not move during the Fantasyland hand**.

Treat the Fantasyland hand as a **continuation hand** tied to the prior hand state for button purposes.

Implementation rule:
- if a hand leads into a Fantasyland continuation hand for one or both players, do **not** rotate the button between those two hands
- once the Fantasyland continuation hand is fully resolved, resume normal button rotation on the next non-continuation hand

---

## 4. Hidden and Public Information

### 4.1 Public information

The following are public:

- all cards that have been placed onto either player's visible board during a normal hand
- row capacities and fill counts
- whose turn it is
- whether a player is currently in Fantasyland for the next hand

### 4.2 Hidden information

The following are hidden:

- the order of the undealt deck
- the current private 3-card Pineapple draw before action is taken
- each player's discarded card on every Pineapple draw round
- Fantasyland-set boards until showdown

### 4.3 Discard visibility rule

Discarded Pineapple cards remain **hidden from the opponent forever**.

They are dead cards and are not returned to the deck, but the opponent never sees them.

### 4.4 Fantasyland concealment rule

A Fantasyland hand is set **concealed until showdown**.

If a player is in Fantasyland, that player's chosen 13-card arrangement is not revealed until hand resolution.

---

## 5. Flow of a Normal Pineapple Hand

A normal Pineapple hand proceeds as follows.

### 5.1 Initial deal

1. The first player to act receives **5 cards**.
2. That player places **all 5 cards** onto their board, respecting row capacities.
3. The other player receives **5 cards**.
4. That player places **all 5 cards** onto their board, respecting row capacities.

### 5.2 Pineapple draw rounds

After both players have placed their initial 5 cards, there are **4 Pineapple draw turns per player**.

On each such turn:

1. the active player receives **3 private cards**
2. the player chooses **exactly 2** of those cards to place onto legal board positions
3. the player chooses **exactly 1** of those 3 cards to discard face down
4. the discarded card is hidden permanently from the opponent
5. once placed, cards can never be moved

The two chosen placement cards may both be placed into the same row if that row has enough remaining capacity.

A player may never place more than the row capacity.

After 4 Pineapple draw turns, each player has placed:
- initial 5 cards
- plus 4 × 2 = 8 later cards
- total = **13 placed cards**

At that point, the hand is complete and goes to showdown.

---

## 6. Fantasyland Rules

## 6.1 Entering Fantasyland

A player enters **Fantasyland for the next hand** if:

- the current hand is **legal**; and
- the player finishes with **QQ or better in the Top row**

For this rule, **QQ or better** means:
- QQ
- KK
- AA
- any trips in the Top row

A fouled hand never enters Fantasyland.

## 6.2 Fantasyland hand structure

In Pineapple Fantasyland:

- the Fantasyland player receives **14 cards at once**
- the player chooses **13** of those cards to place into Top / Middle / Bottom
- the player discards **1**
- the arrangement remains **concealed until showdown**

If only one player is in Fantasyland, the other player plays a normal Pineapple hand.

If both players are in Fantasyland, both players receive 14 cards and both set concealed hands.

## 6.3 Staying in Fantasyland

Use the following exact stay-in-Fantasyland conditions.

A player already in Fantasyland stays in Fantasyland for the next hand if their completed Fantasyland hand is legal and at least one of the following is true:

### Bottom row stay condition
- **Quads** or better

This includes:
- Quads
- Straight Flush
- Royal Flush

### Middle row stay condition
- **Full House** or better

This includes:
- Full House
- Quads
- Straight Flush
- Royal Flush

### Top row stay condition
- **Trips of any rank**

This includes:
- 222 through AAA in the Top row

Unspecified Fantasyland continuation order will be resolved by reusing standard hand turn order. The button remains unchanged, and the player to the left of the button acts first. Fantasyland set actions occur on that player’s turn, even though the hand remains concealed until showdown.

If none of these conditions are met, the player leaves Fantasyland after that hand.

---

## 7. Row Comparison at Showdown

At showdown, compare the corresponding rows:

- Top vs Top
- Middle vs Middle
- Bottom vs Bottom

For each row:
- stronger row earns **+1**
- weaker row earns **-1**
- tied row earns **0**

### 7.1 Sweep bonus

If a player wins all 3 rows, that player gets:
- **+3** additional points

The losing player correspondingly gets:
- **-3** additional points

Thus a full sweep is worth:
- **+6** total from row outcomes plus sweep bonus
- **-6** to the opponent

---

## 8. Royalties / Bonuses

Royalties are awarded only if the player's hand is **legal**, except that royalties **do still count if the opponent fouls**.

A fouled hand receives **no royalties**.

Royalties are added on top of row scoring and sweep scoring.

## 8.1 Bottom row royalties

- Straight = **+2**
- Flush = **+4**
- Full House = **+6**
- Quads = **+10**
- Straight Flush = **+15**
- Royal Flush = **+25**

## 8.2 Middle row royalties

- Three of a Kind = **+2**
- Straight = **+4**
- Flush = **+8**
- Full House = **+12**
- Quads = **+20**
- Straight Flush = **+30**
- Royal Flush = **+50**

## 8.3 Top row royalties

### Pairs
- 66 = **+1**
- 77 = **+2**
- 88 = **+3**
- 99 = **+4**
- TT = **+5**
- JJ = **+6**
- QQ = **+7**
- KK = **+8**
- AA = **+9**

### Trips
- 222 = **+10**
- 333 = **+11**
- 444 = **+12**
- 555 = **+13**
- 666 = **+14**
- 777 = **+15**
- 888 = **+16**
- 999 = **+17**
- TTT = **+18**
- JJJ = **+19**
- QQQ = **+20**
- KKK = **+21**
- AAA = **+22**

---

## 9. Foul Handling and Terminal Scoring

Terminal scoring in this engine is **zero-sum** between the two players.

- if one player's final score is **+X**
- the other player's final score is **-X**

This applies to normal row scoring, sweep bonuses, royalties, and one-player-foul outcomes.

## 9.1 If both players are legal

If both players have legal hands:

- score each row:
  - win = +1
  - tie = 0
  - loss = -1
- add sweep bonus if applicable
- add each player's royalties
- final net score is:

```text
(player's row points)
+ (player's sweep bonus, if any)
+ (player's royalties)
- (opponent's royalties)
```

## 9.2 If one player fouls

If exactly one player fouls:

- the fouling player gets **0 row wins**
- the non-fouling player is treated exactly as though they **swept**
- therefore the non-fouling player gets:
  - **+3** for winning all 3 rows
  - **+3** for the sweep bonus
  - **plus any royalties they earned**
- the fouling player gets:
  - **-3** for losing all 3 rows
  - **-3** for being swept
  - **no royalties**

So the terminal result is:

- legal player: **+6 + legal player's royalties**
- fouling player: **-(6 + legal player's royalties)**

This rule is equivalent to treating the non-fouling player as having completed a standard sweep, and royalties **do count** for the legal player even when the opponent fouls. Because the game is scored zero-sum, the fouling player's final score is the exact negative of the legal player's final score.

## 9.3 If both players foul

If both players foul:

- both players receive **0**
- no row points
- no sweep bonus
- no royalties are awarded to either player

---

## 10. Ties

Use the following tie rules:

- row ties score **0**
- if both players foul, both score **0**
- suits never break ties

There are no special house rules such as:
- button wins ties
- naturals
- surrender
- shoot-the-moon side rules

---

## 11. Engine Requirements / Implementation Constraints

The engine must enforce the following:

1. A card, once placed, cannot be moved.
2. A row cannot exceed capacity.
3. On a normal Pineapple draw turn, the player must:
   - receive exactly 3 cards
   - place exactly 2
   - discard exactly 1
4. Discarded cards are dead and hidden forever.
5. Fantasyland hands are concealed until showdown.
6. A fouled hand earns no royalties and cannot enter or stay in Fantasyland.
7. Fantasyland entry is based on a **legal** hand with **QQ+ on top**.
8. Staying in Fantasyland uses the exact conditions in Section 6.3.
9. The button does not move during a Fantasyland continuation hand.
10. After the Fantasyland continuation hand resolves, normal button rotation resumes on the next non-continuation hand.

---

## 12. Recommended Interpretation for Solver / Engine Interfaces

For implementation purposes, it is recommended to represent:

- variant configuration separately from game state
- public state separately from hidden state
- terminal scoring as a pure function
- legality / foul detection as a pure function
- row comparison as a pure function
- Fantasyland transition logic as a pure function

This will make the engine easier to test and easier to use later for rollouts, search, and solver work.
