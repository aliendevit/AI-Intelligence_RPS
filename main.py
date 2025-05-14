# main.py

LABELS = ["rock", "paper", "scissors"]
wins_set = {("rock", "scissors"), ("paper", "rock"), ("scissors", "paper")}

def decide_winner(player, computer):
    """Determine the outcome of a round."""
    if player == computer:
        return "Draw"
    elif (player, computer) in wins_set:
        return "Player Wins"
    else:
        return "Computer Wins"

def is_bbox_stable(current_bbox, prev_bbox, threshold):
    """Check if hand bounding box has minimal motion (stability)."""
    if prev_bbox is None:
        return False
    dx = abs(current_bbox[0] - prev_bbox[0])
    dy = abs(current_bbox[1] - prev_bbox[1])
    return dx < threshold and dy < threshold
