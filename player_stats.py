# player_stats.py

import json
import os

PLAYER_STATS_PATH = "player_stats.json"
MIN_ROUNDS_FOR_PROFILE = 10

def load_player_stats():
    if os.path.exists(PLAYER_STATS_PATH):
        with open(PLAYER_STATS_PATH, "r") as f:
            return json.load(f)
    else:
        return {}

def save_player_stats(stats):
    with open(PLAYER_STATS_PATH, "w") as f:
        json.dump(stats, f)

player_stats = load_player_stats()

def update_player_stats(player, player_gesture, ai_gesture, outcome):
    if player not in player_stats:
        player_stats[player] = {
            "rounds": 0,
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "gesture_counts": {"rock": 0, "paper": 0, "scissors": 0}
        }
    stats = player_stats[player]
    stats["rounds"] += 1
    stats["gesture_counts"][player_gesture] += 1
    if outcome == "Player Wins":
        stats["wins"] += 1
    elif outcome == "Computer Wins":
        stats["losses"] += 1
    else:
        stats["draws"] += 1
    save_player_stats(player_stats)

def get_player_level_message(player):
    stats = player_stats.get(player, {})
    rounds = stats.get("rounds", 0)
    if rounds < MIN_ROUNDS_FOR_PROFILE:
        return f"Hi {player}, play at least {MIN_ROUNDS_FOR_PROFILE - rounds} more rounds for full analysis!"
    else:
        win_rate = 100 * stats["wins"] / rounds if rounds else 0
        fav_move = max(stats["gesture_counts"], key=stats["gesture_counts"].get)
        return f"{player}: {rounds} rounds, Win rate: {win_rate:.1f}%, Favorite: {fav_move.capitalize()}"
