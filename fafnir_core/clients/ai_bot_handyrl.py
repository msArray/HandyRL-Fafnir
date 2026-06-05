# fafnir_core/clients/ai_bot_handyrl.py
import asyncio
import argparse
import random
import sys
import os
from typing import Any, Dict, List, Optional

# Add workspace root to sys.path so we can import fafnir_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import torch
import socketio

from fafnir_env import FafnirModel, ID_TO_ACTION, is_legal, COLORS_ALL, get_legal_actions

sio = socketio.AsyncClient(reconnection=True, ssl_verify=False)

cfg = {"room": "room1", "name": "AI_HandyRL", "url": "http://127.0.0.1:8765"}

my_index: Optional[int] = None
last_state: Optional[Dict[str, Any]] = None

# AI neural network model
model: Optional[FafnirModel] = None

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
    # Extract details safely
    hand = my_hand(st)
    my_hand_counts = [hand.count(c) for c in COLORS_ALL]
    
    opp_idx = 1 - my_idx
    ps = players_of(st)
    opp_hand_size = ps[opp_idx].get("hand_count", 0) if opp_idx < len(ps) else 0
    
    offer = safe_list(st.get("offer"))
    offer_counts = [offer.count(c) for c in COLORS_ALL]
    
    trash = st.get("trash") or {}
    trash_counts = [trash.get(c, 0) for c in COLORS_ALL]
    
    bag_size = st.get("bag_left", 0)
    
    my_score = ps[my_idx].get("score", 0) if my_idx < len(ps) else 0
    opp_score = ps[opp_idx].get("score", 0) if opp_idx < len(ps) else 0
    
    caretaker = st.get("caretaker", 0)
    is_caretaker = 1.0 if caretaker == my_idx else 0.0
    is_p0 = 1.0 if my_idx == 0 else 0.0
    
    turn_num = st.get("turn", 1)
    round_num = st.get("round", 1)
    
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
         turn_num / 20.0, 
         round_num / 10.0]
    )
    return np.array(features, dtype=np.float32)


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
        # Fallback to empty bid if somehow no actions are legal (should not happen as empty is always legal)
        bid = []
    else:
        # Set illegal action logits to a very low value
        masked_logits = np.ones_like(policy_logits) * -1e32
        masked_logits[legal_indices] = policy_logits[legal_indices]
        
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
    global AUTO_NEXT, model

    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--room", default="room1")
    ap.add_argument("--name", default="AI_HandyRL")
    ap.add_argument("--model", default="models/latest.pth", help="Path to HandyRL trained model pth")
    ap.add_argument("--auto-next", type=int, default=1, help="1=auto OK/Next in RESULT/ROUND_END, 0=disable")
    args = ap.parse_args()

    cfg["url"] = args.url
    cfg["room"] = args.room
    cfg["name"] = args.name

    AUTO_NEXT = bool(args.auto_next)

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
