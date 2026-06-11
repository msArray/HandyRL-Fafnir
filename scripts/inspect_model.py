# scripts/inspect_model.py
import sys
import os
import torch
import numpy as np

# Add workspace root to sys.path so we can import fafnir_env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fafnir_env import FafnirModel, ID_TO_ACTION, get_legal_actions, COLORS_ALL

def main():
    model_path = "latest.pth"
    if len(sys.argv) > 1:
        model_path = sys.argv[1]

    if not os.path.exists(model_path):
        # Check in models/ directory as well
        alt_path = os.path.join("models", model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"Error: Model file '{model_path}' not found.")
            sys.exit(1)

    print(f"=========================================")
    print(f" Loading model: {model_path}")
    print(f"=========================================\n")

    # Load weights
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    except Exception as e:
        print(f"Error loading model weights: {e}")
        sys.exit(1)

    # 1. Print layer statistics
    print("--- 1. Model Weights Statistics ---")
    for key, tensor in state_dict.items():
        weight_np = tensor.numpy()
        mean_val = np.mean(weight_np)
        std_val = np.std(weight_np)
        min_val = np.min(weight_np)
        max_val = np.max(weight_np)
        print(f"Layer: {key:<20} | Shape: {str(list(tensor.shape)):<12} | Mean: {mean_val:8.4f} | Std: {std_val:8.4f} | Range: [{min_val:8.4f}, {max_val:8.4f}]")
    print()

    # 2. Instantiate FafnirModel and load weights
    model = FafnirModel()
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # 3. Simulate an inference to see Policy (p) and Value (v)
    print("--- 2. Example State Inference (Policy 'p' and Value 'v') ---")
    
    # Let's define a sample state:
    # my hand: 3 gold, 1 red, 1 green
    my_hand = ["gold", "gold", "gold", "red", "green"]
    # opponent hand size: 8
    opp_hand_size = 8
    # offer: 1 yellow, 1 blue
    offer = ["yellow", "blue"]
    # trash: 2 gold, 1 red, 3 blue
    trash = {"gold": 2, "red": 1, "orange": 0, "yellow": 0, "green": 0, "blue": 3}
    # bag size: 15
    bag_size = 15
    # my score: 12, opp score: 8
    my_score = 12
    opp_score = 8
    # caretaker: me (1.0), player_idx = 0 (is_p0 = 1.0)
    caretaker = 0
    my_index = 0
    turn_num = 4
    round_num = 2

    # Map state to features (26 features)
    my_hand_counts = [my_hand.count(c) for c in COLORS_ALL]
    offer_counts = [offer.count(c) for c in COLORS_ALL]
    trash_counts = [trash.get(c, 0) for c in COLORS_ALL]
    
    is_caretaker = 1.0 if caretaker == my_index else 0.0
    is_p0 = 1.0 if my_index == 0 else 0.0

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
    
    obs = np.array(features, dtype=np.float32)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    print(f"Sample Input State:")
    print(f"  - My Hand: {my_hand}")
    print(f"  - Opponent Hand Size: {opp_hand_size}")
    print(f"  - Offer: {offer}")
    print(f"  - Trash: {trash}")
    print(f"  - Bag Size: {bag_size}")
    print(f"  - Scores: Me {my_score} vs Opponent {opp_score}")
    print(f"  - Caretaker: {'Yes' if caretaker == my_index else 'No'}")
    print(f"  - Turn: {turn_num}, Round: {round_num}")
    print()

    # Forward pass
    with torch.no_grad():
        outputs = model(obs_t)
        policy_logits = outputs['policy'].squeeze(0).numpy()
        value = outputs['value'].squeeze(0).item()

    # Calculate probabilities of legal actions (Softmax over legal actions)
    legal_indices = get_legal_actions(my_hand, offer)
    print(f"Number of Legal Actions: {len(legal_indices)} / 462")
    
    if not legal_indices:
        print("No legal actions available.")
    else:
        # Softmax over only legal actions
        legal_logits = policy_logits[legal_indices]
        # Subtract max for numerical stability in softmax
        exp_logits = np.exp(legal_logits - np.max(legal_logits))
        legal_probs = exp_logits / np.sum(exp_logits)
        
        # Sort by probability descending
        sorted_idx = np.argsort(legal_probs)[::-1]
        
        print("\nTop 5 predicted actions (p-values / probabilities):")
        for rank in range(min(5, len(legal_indices))):
            idx_in_legal = sorted_idx[rank]
            action_id = legal_indices[idx_in_legal]
            prob = legal_probs[idx_in_legal]
            logit = legal_logits[idx_in_legal]
            action_tuple = ID_TO_ACTION[action_id]
            action_str = ",".join(action_tuple) if action_tuple else "empty"
            print(f"  {rank+1}. Action: {action_str:<25} | Probability: {prob:6.2%} (logit: {logit:6.2f})")

    print(f"\nPredicted State Value (v-value): {value:8.4f} (range: -1.0 [Loss] to +1.0 [Win])")
    print(f"=========================================")

if __name__ == "__main__":
    main()
