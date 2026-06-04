# fafnir_env.py
import itertools
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from handyrl.environment import BaseEnvironment

# =========================
# Constants
# =========================
COLORS = ["red", "orange", "yellow", "green", "blue"]  # priority order for tie-break
GOLD = "gold"
COLORS_ALL = ["gold", "red", "orange", "yellow", "green", "blue"]
STONE_COUNT = {GOLD: 20, "red": 12, "orange": 12, "yellow": 12, "green": 12, "blue": 12}

TRASH_LIMIT = 6
SEED_TRASH_AT_ROUND_START = 3

POINT_CHIP = 1
SCORE_TO_WIN = 40

# =========================
# Action Space Mapping
# =========================
def generate_action_map(max_size=5):
    actions = []
    for r in range(max_size + 1):
        for comb in itertools.combinations_with_replacement(COLORS_ALL, r):
            actions.append(comb)
    action_to_id = {comb: i for i, comb in enumerate(actions)}
    id_to_action = {i: comb for i, comb in enumerate(actions)}
    return action_to_id, id_to_action

ACTION_TO_ID, ID_TO_ACTION = generate_action_map(5)

def is_legal(action_tuple, hand, offer):
    for color in action_tuple:
        if color in offer:
            return False
    for color in set(action_tuple):
        if hand.count(color) < action_tuple.count(color):
            return False
    return True

# =========================
# Neural Network Model
# =========================
class FafnirModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input size: 26 features
        self.fc1 = nn.Linear(26, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        
        # Policy output: 462 actions
        self.policy_head = nn.Linear(128, 462)
        # Value output: 1 scalar
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, hidden=None):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        
        policy = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        
        return {'policy': policy, 'value': value}

# =========================
# HandyRL Environment Wrapper
# =========================
class Environment(BaseEnvironment):
    def __init__(self, args={}):
        super().__init__()
        self.action_to_id = ACTION_TO_ID
        self.id_to_action = ID_TO_ACTION
        self.reset()

    def reset(self, args={}):
        self.bag = self.make_bag()
        self.trash = {c: 0 for c in STONE_COUNT.keys()}
        self.trash_pile = []
        self.offer = []
        
        self.players_state = [
            {"stones": [], "score": 0},
            {"stones": [], "score": 0}
        ]
        self.round = 1
        self.turn_num = 1
        self.caretaker = random.randint(0, 1)
        self.current_player = 0
        self.bids = {0: [], 1: []}
        
        self.win_player = None
        self.record = []
        
        self.deal_initial_hands()
        self.seed_trash_at_round_start()
        self.setup_offer()
        
        return None

    def make_bag(self):
        bag = []
        for c, n in STONE_COUNT.items():
            bag.extend([c] * n)
        random.shuffle(bag)
        return bag

    def draw_one(self):
        if not self.bag:
            return None
        return self.bag.pop()

    def deal_initial_hands(self):
        for i, p in enumerate(self.players_state):
            p["stones"] = []
        for i, p in enumerate(self.players_state):
            n = 11 if i == self.caretaker else 10
            for _ in range(n):
                s = self.draw_one()
                if s is not None:
                    p["stones"].append(s)

    def seed_trash_at_round_start(self):
        for _ in range(SEED_TRASH_AT_ROUND_START):
            s = self.draw_one()
            if s is None:
                break
            self.trash[s] += 1
            self.trash_pile.append(s)

    def setup_offer(self):
        self.offer = []
        if len(self.bag) == 0:
            return False
        
        stones = []
        random.shuffle(self.bag)
        while True:
            draw_n = min(2, len(self.bag))
            for _ in range(draw_n):
                s = self.draw_one()
                if s is not None:
                    stones.append(s)
            
            if len(set(stones)) > 1 or len(self.bag) == 0:
                break
        self.offer = stones
        return bool(self.offer)

    def line_up_trash(self, stones):
        for s in stones:
            self.trash[s] += 1
        self.trash_pile.extend(stones)

    def rank_colors_by_total_in_hands(self):
        totals = {c: 0 for c in COLORS}
        for p in self.players_state:
            for s in p["stones"]:
                if s in COLORS:
                    totals[s] += 1
        priority = list(COLORS)
        return sorted(totals.items(), key=lambda x: (-x[1], priority.index(x[0])))

    def compute_round_scores(self):
        ranked = self.rank_colors_by_total_in_hands()
        first = ranked[0][0] if ranked else None
        second = ranked[1][0] if len(ranked) > 1 else None
        
        adds = []
        for p in self.players_state:
            score = 0
            score += p["stones"].count(GOLD)
            for c in COLORS:
                cnt = p["stones"].count(c)
                if cnt == 0:
                    continue
                if cnt >= 5:
                    continue
                
                if c == first:
                    mult = 3
                elif c == second:
                    mult = 2
                else:
                    mult = -1
                score += cnt * mult
            adds.append(score)
        return ranked, adds

    def process_round_end(self):
        ranked, adds = self.compute_round_scores()
        for i, add in enumerate(adds):
            self.players_state[i]["score"] = max(0, self.players_state[i]["score"] + add)
            
        if self.check_game_end_local():
            return
            
        self.bag = self.make_bag()
        self.trash = {c: 0 for c in STONE_COUNT.keys()}
        self.trash_pile = []
        self.offer = []
        
        self.deal_initial_hands()
        self.seed_trash_at_round_start()
        self.setup_offer()
        
        self.round += 1
        self.turn_num = 1
        self.current_player = 0

    def check_game_end_local(self):
        for i, p in enumerate(self.players_state):
            if p["score"] >= SCORE_TO_WIN:
                self.win_player = i
                return True
        return False

    def play(self, action_id, player=None):
        if player is None:
            player = self.current_player
        
        self.record.append(action_id)
        action_tuple = self.id_to_action[action_id]
        self.bids[player] = list(action_tuple)
        
        if player == 0:
            self.current_player = 1
        else:
            self.resolve_auction()

    def resolve_auction(self):
        bids_p0 = self.bids[0]
        bids_p1 = self.bids[1]
        
        bids_count = [len(bids_p0), len(bids_p1)]
        max_bid = max(bids_count)
        
        if max_bid == 0:
            for p in self.players_state:
                p["score"] = max(0, p["score"] - 1)
            self.line_up_trash(self.offer)
            self.offer = []
            winner = None
        else:
            candidates = [i for i, b in enumerate(bids_count) if b == max_bid]
            if len(candidates) == 1:
                winner = candidates[0]
            else:
                ct = self.caretaker
                if ct in candidates:
                    non_ct = [i for i in candidates if i != ct]
                    winner = min(non_ct) if non_ct else ct
                else:
                    winner = min(candidates)
            
            used = self.bids[winner]
            for s in used:
                try:
                    self.players_state[winner]["stones"].remove(s)
                except ValueError:
                    pass
            if used:
                self.line_up_trash(used)
            
            if self.offer:
                self.players_state[winner]["stones"].extend(self.offer)
            self.offer = []
            
            self.players_state[winner]["score"] += POINT_CHIP
            self.caretaker = winner
            
        self.bids = {0: [], 1: []}
        
        if self.check_game_end_local():
            return
            
        bag_low = len(self.bag) < 2
        trash_limit_reached = any(self.trash[c] >= TRASH_LIMIT for c in self.trash)
        
        if bag_low or trash_limit_reached:
            self.process_round_end()
        else:
            ok = self.setup_offer()
            if not ok or len(self.bag) < 2:
                self.process_round_end()
            else:
                self.turn_num += 1
                self.current_player = 0

    def turn(self):
        return self.current_player

    def terminal(self):
        return self.win_player is not None

    def outcome(self):
        if self.win_player is None:
            return {0: 0, 1: 0}
        outcomes = {}
        for p in [0, 1]:
            if p == self.win_player:
                outcomes[p] = 1
            else:
                outcomes[p] = -1
        return outcomes

    def legal_actions(self, player=None):
        if player is None:
            player = self.current_player
        
        hand = self.players_state[player]["stones"]
        offer = self.offer
        
        legal_ids = []
        for action_id, action_tuple in self.id_to_action.items():
            if is_legal(action_tuple, hand, offer):
                legal_ids.append(action_id)
        
        return legal_ids

    def players(self):
        return [0, 1]

    def observation(self, player=None):
        if player is None:
            player = self.current_player
        opponent = 1 - player
        
        my_hand = self.players_state[player]["stones"]
        my_hand_counts = [my_hand.count(c) for c in COLORS_ALL]
        
        opp_hand_size = len(self.players_state[opponent]["stones"])
        
        offer_counts = [self.offer.count(c) for c in COLORS_ALL]
        
        trash_counts = [self.trash[c] for c in COLORS_ALL]
        
        bag_size = len(self.bag)
        
        my_score = self.players_state[player]["score"]
        opp_score = self.players_state[opponent]["score"]
        
        is_caretaker = 1.0 if self.caretaker == player else 0.0
        is_p0 = 1.0 if player == 0 else 0.0
        
        features = (
            my_hand_counts + 
            [opp_hand_size / 20.0] + 
            offer_counts + 
            [tc / 6.0 for tc in trash_counts] + 
            [bag_size / 80.0, 
             my_score / 40.0, 
             opp_score / 40.0, 
             is_caretaker, 
             is_p0, 
             self.turn_num / 20.0, 
             self.round / 10.0]
        )
        
        return np.array(features, dtype=np.float32)

    def action2str(self, action_id, player=None):
        action_tuple = self.id_to_action[action_id]
        return ",".join(action_tuple) if action_tuple else "empty"

    def str2action(self, s, player=None):
        if s == "empty" or s == "":
            action_tuple = ()
        else:
            action_tuple = tuple(s.split(","))
        return self.action_to_id[action_tuple]

    def diff_info(self, player=None):
        if not self.record:
            return ""
        return self.action2str(self.record[-1])

    def update(self, info, reset):
        if reset:
            self.reset()
        else:
            action = self.str2action(info)
            self.play(action, self.current_player)

    def net(self):
        return FafnirModel()
