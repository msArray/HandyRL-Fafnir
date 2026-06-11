# fafnir_core/clients/ai_bot_handyrl_v3.py
import asyncio
import argparse
import random
import sys
import os
import time
import copy
from typing import Any, Dict, List, Optional

# Add workspace root to sys.path so we can import fafnir_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
import socketio

from fafnir_env import (
    Environment,
    FafnirModel,
    ID_TO_ACTION,
    ACTION_TO_ID,
    COLORS,
    COLORS_ALL,
    STONE_COUNT,
    get_legal_actions,
)

sio = socketio.AsyncClient(reconnection=True, ssl_verify=False)

cfg = {"room": "room1", "name": "AI_HandyRL_V3", "url": "http://127.0.0.1:8765"}

my_index: Optional[int] = None
last_state: Optional[Dict[str, Any]] = None
known_stones = [{c: 0 for c in COLORS_ALL}, {c: 0 for c in COLORS_ALL}]
known_round: Optional[int] = None
known_result_key: Optional[str] = None

# AI neural network model
model: Optional[FafnirModel] = None

# anti-spam
_action_lock = asyncio.Lock()
_last_emit_ts = 0.0

# OK debounce per RESULT/ROUND_END instance
_ok_sent_key: Optional[str] = None

AUTO_NEXT = True
THINK_TIME = 0.2  # Dynamic thinking time budget in seconds
WEIGHT = 0.5      # Weight for simulated value head


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


def sample_environment(st: Dict[str, Any], my_idx: int, known_stones_list: List[Dict[str, int]]) -> Environment:
    env = Environment()
    opp_idx = 1 - my_idx

    # 1. Gather known stones
    my_h = my_hand(st)
    offer = safe_list(st.get("offer"))
    trash = st.get("trash") or {}

    # Opponent's known hand
    opp_k = known_stones_list[opp_idx]
    opp_k_list = []
    for c, cnt in opp_k.items():
        opp_k_list.extend([c] * cnt)

    # Accumulate all known stones in play/discarded
    all_known = {c: 0 for c in COLORS_ALL}
    for c in my_h:
        all_known[c] += 1
    for c in offer:
        all_known[c] += 1
    for c, cnt in trash.items():
        all_known[c] += cnt
    for c in opp_k_list:
        all_known[c] += 1

    # 2. Reconstruct the unknown stones pool
    pool = []
    for c, total_limit in STONE_COUNT.items():
        remaining = total_limit - all_known[c]
        if remaining > 0:
            pool.extend([c] * remaining)

    # Opponent's hidden hand size
    ps = players_of(st)
    opp_hand_size = ps[opp_idx].get("hand_count", 0) if opp_idx < len(ps) else 0
    opp_unknown_size = max(0, opp_hand_size - len(opp_k_list))

    # Sample opponent's unknown stones
    random.shuffle(pool)
    opp_sampled_hand = opp_k_list.copy()
    if opp_unknown_size > 0:
        sampled = pool[:opp_unknown_size]
        opp_sampled_hand.extend(sampled)
        pool = pool[opp_unknown_size:]

    # Remaining stones form the bag
    bag = pool
    random.shuffle(bag)

    # 3. Populate simulator state
    env.bag = bag
    env.trash = {c: trash.get(c, 0) for c in COLORS_ALL}
    env.trash_pile = []
    for c, cnt in env.trash.items():
        env.trash_pile.extend([c] * cnt)

    env.offer = offer.copy()

    p0_hand = my_h if my_idx == 0 else opp_sampled_hand
    p1_hand = my_h if my_idx == 1 else opp_sampled_hand

    p0_score = ps[0].get("score", 0) if len(ps) > 0 else 0
    p1_score = ps[1].get("score", 0) if len(ps) > 1 else 0

    env.players_state = [
        {"stones": p0_hand.copy(), "score": p0_score},
        {"stones": p1_hand.copy(), "score": p1_score}
    ]

    try:
        env.round = int(st.get("round", 1))
        env.turn_num = int(st.get("turn", 1))
    except Exception:
        env.round = 1
        env.turn_num = 1

    env.caretaker = int(st.get("caretaker", 0))
    env.current_player = my_idx
    env.bids = {0: [], 1: []}
    env.known_stones = [dict(known_stones_list[0]), dict(known_stones_list[1])]
    env.win_player = None

    # Track scores for immediate reward calculation
    env.prev_scores = [p0_score, p1_score]
    env.rewards = {0: 0.0, 1: 0.0}

    return env


def clone_env(env: Environment) -> Environment:
    new_env = Environment()
    new_env.bag = env.bag.copy()
    new_env.trash = env.trash.copy()
    new_env.trash_pile = env.trash_pile.copy()
    new_env.offer = env.offer.copy()
    new_env.players_state = [
        {"stones": p["stones"].copy(), "score": p["score"]} for p in env.players_state
    ]
    new_env.round = env.round
    new_env.turn_num = env.turn_num
    new_env.caretaker = env.caretaker
    new_env.current_player = env.current_player
    new_env.bids = {0: env.bids[0].copy(), 1: env.bids[1].copy()}
    new_env.known_stones = [env.known_stones[0].copy(), env.known_stones[1].copy()]
    new_env.win_player = env.win_player
    new_env.prev_scores = env.prev_scores.copy()
    new_env.rewards = env.rewards.copy()
    return new_env


def state_to_observation(st: Dict[str, Any], my_idx: int) -> np.ndarray:
    # Extract details safely
    hand = my_hand(st)
    my_hand_counts = [hand.count(c) / float(STONE_COUNT[c]) for c in COLORS_ALL]
    
    opp_idx = 1 - my_idx
    ps = players_of(st)
    opp_hand_size = ps[opp_idx].get("hand_count", 0) if opp_idx < len(ps) else 0
    opp_known = known_stones[opp_idx]
    my_known = known_stones[my_idx]
    
    offer = safe_list(st.get("offer"))
    offer_counts = [offer.count(c) / 10.0 for c in COLORS_ALL]
    
    trash = st.get("trash") or {}
    trash_counts = [trash.get(c, 0) for c in COLORS_ALL]
    
    bag_size = st.get("bag_left", 0)

    is_caretaker = 1.0 if st.get("caretaker", 0) == my_idx else 0.0
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


def select_action_with_search(st: Dict[str, Any], my_idx: int, think_time: float) -> List[str]:
    hand = my_hand(st)
    offer = safe_list(st.get("offer"))
    legal_action_ids = get_legal_actions(hand, offer)

    if not legal_action_ids:
        return []

    # If only one legal move exists, return it immediately to save time
    if len(legal_action_ids) == 1:
        return list(ID_TO_ACTION[legal_action_ids[0]])

    # 1. Compute Policy Softmax at the current state (Root)
    root_obs = state_to_observation(st, my_idx)
    root_obs_t = torch.tensor(root_obs, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        root_outputs = model(root_obs_t)
        root_logits = root_outputs['policy'].squeeze(0).numpy()
        
    masked_logits = np.ones_like(root_logits) * -1e32
    masked_logits[legal_action_ids] = root_logits[legal_action_ids]
    exp_logits = np.exp(masked_logits - np.max(masked_logits[legal_action_ids]))
    policy_probs = exp_logits / np.sum(exp_logits)

    opp_idx = 1 - my_idx
    accum_scores = {aid: 0.0 for aid in legal_action_ids}
    sim_counts = {aid: 0 for aid in legal_action_ids}

    start_time = time.time()
    runs = 0

    while time.time() - start_time < think_time:
        # 1-1. Sample state
        env = sample_environment(st, my_idx, known_stones)

        # 1-2. Predict opponent's action (Stochastic Sampling from opponent's policy)
        opp_obs = env.observation(opp_idx)
        opp_obs_t = torch.tensor(opp_obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            opp_outputs = model(opp_obs_t)
            opp_logits = opp_outputs['policy'].squeeze(0).numpy()

        opp_legal = env.legal_actions(opp_idx)
        if not opp_legal:
            opp_action_id = ACTION_TO_ID[()]  # empty bid
        else:
            opp_masked = np.ones_like(opp_logits) * -1e32
            opp_masked[opp_legal] = opp_logits[opp_legal]
            opp_exp = np.exp(opp_masked - np.max(opp_masked[opp_legal]))
            opp_probs = opp_exp / np.sum(opp_exp)
            
            # Stochastic Choice
            choices = np.arange(len(opp_legal))
            probs_subset = opp_probs[opp_legal]
            probs_subset = probs_subset / np.sum(probs_subset)  # re-normalize for rounding errors
            sampled_idx = np.random.choice(choices, p=probs_subset)
            opp_action_id = opp_legal[sampled_idx]

        # 1-3. Simulate all my legal actions in parallel (batched inference)
        sim_envs = []
        for my_action_id in legal_action_ids:
            sim_env = clone_env(env)
            sim_env.play(my_action_id, my_idx)
            sim_env.play(opp_action_id, opp_idx)
            sim_envs.append(sim_env)

        # Prepare batch observations
        obs_list = [sim_env.observation(my_idx) for sim_env in sim_envs]
        obs_batch = torch.tensor(np.array(obs_list), dtype=torch.float32)

        with torch.no_grad():
            sim_outputs = model(obs_batch)
            values = sim_outputs['value'].squeeze(1).numpy()

        # Accumulate results
        for i, my_action_id in enumerate(legal_action_ids):
            val = values[i]
            sim_env = sim_envs[i]
            if sim_env.terminal():
                outcome = sim_env.outcome()
                val = outcome[my_idx] * 2.0  # boost final outcome signal

            accum_scores[my_action_id] += val
            sim_counts[my_action_id] += 1

        runs += 1

    # 4. Hybrid Score Selection Formula: Score = PolicyProb + WEIGHT * AverageValue
    best_action_id = None
    best_score = -999.0
    best_avg_val = 0.0
    for aid in legal_action_ids:
        if sim_counts[aid] > 0:
            avg_val = accum_scores[aid] / sim_counts[aid]
            score = policy_probs[aid] + WEIGHT * avg_val
            if score > best_score:
                best_score = score
                best_avg_val = avg_val
                best_action_id = aid

    if best_action_id is None:
        best_action_id = legal_action_ids[0]

    best_action_tuple = ID_TO_ACTION[best_action_id]
    print(f"[AI] Search stats: simulations={runs}, root_prob={policy_probs[best_action_id]:.4f}, "
          f"sim_val={best_avg_val:.4f}, score={best_score:.4f}, choice={best_action_tuple}")
    return list(best_action_tuple)


async def do_submit_bid(st: Dict[str, Any], reason: str):
    if model is None:
        print("[AI] Error: Model is not loaded!")
        return

    # Call the hybrid lookahead search
    bid = select_action_with_search(st, my_index, THINK_TIME)

    # Throttled send
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

    # 0.05s delay to avoid instantaneous clicks
    await asyncio.sleep(0.05)
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
    global AUTO_NEXT, model, THINK_TIME, WEIGHT

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--room", default="room1")
    ap.add_argument("--name", default="AI_HandyRL_V3")
    ap.add_argument("--model", default="better-model.pth", help="Path to HandyRL trained model pth")
    ap.add_argument("--auto-next", type=int, default=1, help="1=auto OK/Next in RESULT/ROUND_END, 0=disable")
    ap.add_argument("--think-time", type=float, default=0.2, help="Thinking time budget in seconds")
    ap.add_argument("--weight", type=float, default=0.5, help="Weight for simulation value (default 0.5)")
    args = ap.parse_args()

    cfg["url"] = args.url
    cfg["room"] = args.room
    cfg["name"] = args.name

    AUTO_NEXT = bool(args.auto_next)
    THINK_TIME = args.think_time
    WEIGHT = args.weight

    # Initialize and load model weights
    print(f"Loading HandyRL model from {args.model}...")
    model = FafnirModel()
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
