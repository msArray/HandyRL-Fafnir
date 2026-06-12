# fafnir_core/clients/ai_bot_handyrl.py
import asyncio
import argparse
import random
import sys
import os
import itertools
from typing import Any, Dict, List, Optional

# Add workspace root to sys.path so we can import fafnir_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import socketio

from fafnir_env import (
    COLORS,
    COLORS_ALL,
    STONE_COUNT,
    get_legal_actions,
)

sio = socketio.AsyncClient(reconnection=True, ssl_verify=False)

cfg = {"room": "room1", "name": "AI_HandyRL", "url": "http://127.0.0.1:8765"}

my_index: Optional[int] = None
last_state: Optional[Dict[str, Any]] = None
known_stones = [{c: 0 for c in COLORS_ALL}, {c: 0 for c in COLORS_ALL}]
known_round: Optional[int] = None
known_result_key: Optional[str] = None

# Action space dynamic mapping placeholders
ACTION_TO_ID = {}
ID_TO_ACTION = {}
NUM_ACTIONS = 462
FEATURES_DIM = 34


class FafnirModelCustom(nn.Module):
    def __init__(self, obs_size=34, num_actions=462):
        super().__init__()
        self.fc1 = nn.Linear(obs_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.policy_head = nn.Linear(128, num_actions)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x, hidden=None):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        policy = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return {'policy': policy, 'value': value}


# AI neural network model placeholder
model: Optional[FafnirModelCustom] = None

# anti-spam
_action_lock = asyncio.Lock()
_last_emit_ts = 0.0

# OK debounce per RESULT/ROUND_END instance
_ok_sent_key: Optional[str] = None

AUTO_NEXT = True
THINK_DELAY = 0.05


def _loop_time() -> float:
    return asyncio.get_running_loop().time()


def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def empty_color_counts() -> Dict[str, int]:
    return {c: 0 for c in COLORS_ALL}


def count_stones(stones: List[Any]) -> Dict[str, int]:
    counts = empty_color_counts()
    for s in stones:
        if s in counts:
            counts[s] += 1
    return counts


def setup_action_space(max_bid_size: int):
    global ACTION_TO_ID, ID_TO_ACTION, NUM_ACTIONS
    actions = []
    for r in range(max_bid_size + 1):
        for comb in itertools.combinations_with_replacement(COLORS_ALL, r):
            actions.append(comb)
    ACTION_TO_ID = {comb: i for i, comb in enumerate(actions)}
    ID_TO_ACTION = {i: comb for i, comb in enumerate(actions)}
    NUM_ACTIONS = len(ID_TO_ACTION)


def public_potential_score(hand: List[str], opponent_known: Dict[str, int]) -> int:
    totals = {c: hand.count(c) + int(opponent_known.get(c, 0)) for c in COLORS}
    ranked = sorted(totals.items(), key=lambda x: (-x[1], COLORS.index(x[0])))
    first = ranked[0][0] if ranked else None
    second = ranked[1][0] if len(ranked) > 1 else None

    score = hand.count("gold")
    for c in COLORS:
        cnt = hand.count(c)
        if cnt == 0 or cnt >= 5:
            continue
        if c == first:
            score += cnt * 3
        elif c == second:
            score += cnt * 2
        else:
            score -= cnt
    return score


def update_known_stones_from_state(st: Dict[str, Any]):
    global known_stones, known_round, known_result_key

    try:
        round_num = int(st.get("round", 1))
    except Exception:
        round_num = 1
    if known_round != round_num:
        known_stones = [empty_color_counts(), empty_color_counts()]
        known_round = round_num
        known_result_key = None

    lr = st.get("last_result") or {}
    if not isinstance(lr, dict):
        return

    winner = lr.get("winner")
    loser = lr.get("loser")
    try:
        winner = int(winner)
        loser = int(loser)
    except Exception:
        return
    if winner not in (0, 1) or loser not in (0, 1):
        return

    winner_bid = safe_list(lr.get("winner_bid"))
    loser_bid = safe_list(lr.get("loser_bid"))
    offer = safe_list(lr.get("offer"))
    result_key = (
        f"{round_num}:{winner}:{loser}:"
        f"{tuple(winner_bid)}:{tuple(loser_bid)}:{tuple(offer)}:{lr.get('bids_count')}"
    )
    if known_result_key == result_key:
        return

    if not winner_bid and not loser_bid:
        known_result_key = result_key
        return

    winner_bid_counts = count_stones(winner_bid)
    loser_bid_counts = count_stones(loser_bid)
    offer_counts = count_stones(offer)
    for c in COLORS_ALL:
        known_stones[winner][c] = max(
            0,
            known_stones[winner][c] + offer_counts[c] - winner_bid_counts[c],
        )
        known_stones[loser][c] = max(known_stones[loser][c], loser_bid_counts[c])
    known_result_key = result_key


def phase_of(st: Dict[str, Any]) -> str:
    return str(st.get("phase") or "WAITING")


def current_bidder(st: Dict[str, Any]) -> Optional[int]:
    cb = st.get("current_bidder", None)
    try:
        return int(cb)
    except Exception:
        return None


def players_of(st: Dict[str, Any]) -> List[Dict[str, Any]]:
    ps = st.get("players")
    return ps if isinstance(ps, list) else []


def me_view(st: Dict[str, Any]) -> Dict[str, Any]:
    ps = players_of(st)
    if my_index is None or my_index < 0 or my_index >= len(ps):
        return {}
    v = ps[my_index]
    return v if isinstance(v, dict) else {}


def my_hand(st: Dict[str, Any]) -> List[str]:
    hand = me_view(st).get("hand")
    return [x for x in hand] if isinstance(hand, list) else []


def my_bid_submitted(st: Dict[str, Any]) -> bool:
    return bool(me_view(st).get("bid_submitted", False))


def my_ok_ready(st: Dict[str, Any]) -> bool:
    return bool(me_view(st).get("ok_ready", False))


async def _emit_throttled(event: str, payload: Dict[str, Any], min_interval: float = 0.12):
    global _last_emit_ts
    async with _action_lock:
        dt = _loop_time() - _last_emit_ts
        if dt < min_interval:
            await asyncio.sleep(min_interval - dt)
        _last_emit_ts = _loop_time()
        await sio.emit(event, payload)


def _phase_key(st: Dict[str, Any]) -> str:
    ph = phase_of(st)
    r = st.get("round", "?")
    t = st.get("turn", "?")
    if ph == "ROUND_END":
        return f"ROUND_END:r{r}"
    if ph == "RESULT":
        return f"RESULT:r{r}:t{t}"
    return f"{ph}:r{r}:t{t}"


def state_to_observation(st: Dict[str, Any], my_idx: int) -> np.ndarray:
    hand = my_hand(st)
    my_hand_counts = [hand.count(c) / float(STONE_COUNT[c]) for c in COLORS_ALL]
    
    opp_idx = 1 - my_idx
    ps = players_of(st)
    opp_hand_size = ps[opp_idx].get("hand_count", 0) if opp_idx < len(ps) else 0
    
    offer = safe_list(st.get("offer"))
    offer_counts = [offer.count(c) / 10.0 for c in COLORS_ALL]
    
    trash = st.get("trash") or {}
    trash_counts = [trash.get(c, 0) for c in COLORS_ALL]
    
    bag_size = st.get("bag_left", 0)
    is_caretaker = 1.0 if st.get("caretaker", 0) == my_idx else 0.0

    if FEATURES_DIM == 26:
        # Legacy 26-dimensional feature representation
        my_score = ps[my_idx].get("score", 0) if my_idx < len(ps) else 0
        opp_score = ps[opp_idx].get("score", 0) if opp_idx < len(ps) else 0
        is_p0 = 1.0 if my_idx == 0 else 0.0
        turn_num = st.get("turn", 1)
        round_num = st.get("round", 1)
        
        my_hand_counts_raw = [hand.count(c) for c in COLORS_ALL]
        offer_counts_raw = [offer.count(c) for c in COLORS_ALL]
        trash_counts_raw = [trash.get(c, 0) for c in COLORS_ALL]
        
        features = (
            my_hand_counts_raw + 
            [opp_hand_size / 20.0] + 
            offer_counts_raw + 
            [tc / 6.0 for tc in trash_counts_raw] + 
            [bag_size / 80.0, 
             my_score / 40.0, 
             opp_score / 40.0, 
             is_caretaker, 
             is_p0, 
             turn_num / 20.0, 
             round_num / 10.0]
        )
        return np.array(features, dtype=np.float32)
    else:
        # Current 34-dimensional feature representation
        opp_known = known_stones[opp_idx]
        my_known = known_stones[my_idx]
        opp_unknown_size = max(0, int(opp_hand_size or 0) - sum(opp_known.values()))
        potential_score = public_potential_score(hand, opp_known)
        
        features = (
            my_hand_counts + 
            offer_counts + 
            [tc / 6.0 for tc in trash_counts] + 
            [opp_known[c] / float(STONE_COUNT[c]) for c in COLORS_ALL] +
            [opp_unknown_size / 20.0] +
            [my_known[c] / float(STONE_COUNT[c]) for c in COLORS_ALL] +
            [bag_size / 80.0, 
             is_caretaker, 
             (potential_score + 15.0) / 75.0]
        )
        return np.clip(np.array(features, dtype=np.float32), 0.0, 1.0)


async def do_submit_bid(st: Dict[str, Any], reason: str):
    if model is None:
        print("[AI] Error: Model is not loaded!")
        return
        
    hand = my_hand(st)
    offer = safe_list(st.get("offer"))
    
    # 1. Translate state to network observation representation
    obs = state_to_observation(st, my_index)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    
    # 2. Model inference
    with torch.no_grad():
        outputs = model(obs_t)
        policy_logits = outputs['policy'].squeeze(0).numpy()
        
    # 3. Mask illegal actions
    legal_indices = get_legal_actions(hand, offer)
            
    if not legal_indices:
        bid = []
    else:
        # Filter action IDs that are valid under the bot's custom action space
        # Some global legal actions might be size 6-8, which this model cannot represent
        bot_legal_indices = [idx for idx in legal_indices if idx < NUM_ACTIONS]
        
        if not bot_legal_indices:
            bid = []
        else:
            masked_logits = np.ones_like(policy_logits) * -1e32
            masked_logits[bot_legal_indices] = policy_logits[bot_legal_indices]
            
            best_action_id = np.argmax(masked_logits)
            best_action_tuple = ID_TO_ACTION[best_action_id]
            bid = list(best_action_tuple)

    await asyncio.sleep(THINK_DELAY)
    await _emit_throttled("submit_bid", {"room_id": cfg["room"], "stones": bid})
    print(f"[AI] submit ({reason}) stones={bid}")


async def do_ok_next(st: Dict[str, Any], reason: str):
    global _ok_sent_key
    key = _phase_key(st)
    if _ok_sent_key == key:
        return
    if my_ok_ready(st):
        _ok_sent_key = key
        return

    await asyncio.sleep(THINK_DELAY)
    await _emit_throttled("proceed_phase", {"room_id": cfg["room"]})
    _ok_sent_key = key
    print(f"[AI] OK/Next ({reason})")


async def brain_loop():
    while True:
        st = last_state
        if st and my_index is not None and my_index >= 0:
            ph = phase_of(st)

            if ph == "BIDDING":
                cb = current_bidder(st)
                if cb == my_index and (not my_bid_submitted(st)):
                    await do_submit_bid(st, reason="brain_loop")

            elif ph in ("RESULT", "ROUND_END"):
                if AUTO_NEXT:
                    await do_ok_next(st, reason="brain_loop")

        await asyncio.sleep(0.10)


# ============ socket handlers ============

@sio.event
async def connect():
    print("[AI] connected")
    await _emit_throttled(
        "join_room",
        {"room_id": cfg["room"], "player_name": cfg["name"]},
        min_interval=0.0,
    )


@sio.event
async def disconnect():
    print("[AI] disconnected")


@sio.on("player_assigned")
async def player_assigned(data):
    global my_index
    try:
        my_index = int(data.get("index"))
    except Exception:
        my_index = None
    print("[AI] assigned index =", my_index)


@sio.on("state_update")
async def state_update(state):
    global last_state, _ok_sent_key
    last_state = state
    update_known_stones_from_state(state)

    # reset OK debounce when leaving RESULT/ROUND_END
    ph = phase_of(state)
    if ph not in ("RESULT", "ROUND_END"):
        _ok_sent_key = None


@sio.on("bid_rejected")
async def bid_rejected(data):
    reason = data.get("reason") or ""
    msg = data.get("message") or reason
    print("[AI] BID REJECTED:", msg)


# ============ main ============

async def main():
    global AUTO_NEXT, model, FEATURES_DIM

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--room", default="room1")
    ap.add_argument("--name", default="AI_HandyRL")
    ap.add_argument("--model", default="better-model.pth", help="Path to HandyRL trained model pth")
    ap.add_argument("--auto-next", type=int, default=1, help="1=auto OK/Next in RESULT/ROUND_END, 0=disable")
    ap.add_argument("--max-bid-size", type=int, default=8, help="Bidding range size limit (e.g. 5 or 8)")
    ap.add_argument("--features-dim", type=int, default=34, help="Features size input dimension (26 or 34)")
    args = ap.parse_args()

    cfg["url"] = args.url
    cfg["room"] = args.room
    cfg["name"] = args.name

    AUTO_NEXT = bool(args.auto_next)
    FEATURES_DIM = args.features_dim

    # Dynamically setup action mappings
    setup_action_space(args.max_bid_size)
    print(f"Configured action space with limit={args.max_bid_size} (NUM_ACTIONS={NUM_ACTIONS})")

    # Initialize and load model weights
    print(f"Loading HandyRL model from {args.model}...")
    model = FafnirModelCustom(obs_size=FEATURES_DIM, num_actions=NUM_ACTIONS)
    try:
        model.load_state_dict(torch.load(args.model, map_location=torch.device('cpu')), strict=False)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load model weights: {e}. Model will use random initialization.")

    task_brain = None
    try:
        await sio.connect(cfg["url"], wait_timeout=15)
        task_brain = asyncio.create_task(brain_loop())
        await sio.wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        if task_brain:
            task_brain.cancel()
            try:
                await task_brain
            except Exception:
                pass
        try:
            if sio.connected:
                await sio.disconnect()
        except Exception:
            pass
        # extra safety close
        try:
            eio = getattr(sio, "eio", None)
            if eio is not None and getattr(eio, "connected", False):
                await eio.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
