import numpy as np

def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    # Initialize the distance matrix
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def calculate_ned(predicted, ground_truth):
    """Calculate the Normalized Edit Distance (NED) for a list of predictions and ground truths."""
    total_distance = 0
    for p, gt in zip(predicted, ground_truth):
        dist = levenshtein_distance(p, gt)
        total_distance += dist / max(len(p), len(gt))
    return total_distance / len(predicted)

def calculate_wra(predicted, ground_truth):
    """Calculate the Word Recognition Accuracy (WRA)."""
    correct_words = sum(p == gt for p, gt in zip(predicted, ground_truth))
    return correct_words / len(predicted)

def evaluate_metrics(predicted, ground_truth):
    """Evaluate both WRA and character-level accuracy using NED."""
    ned = calculate_ned(predicted, ground_truth)
    wra = calculate_wra(predicted, ground_truth)
    character_accuracy = 1 - ned
    
    return wra, character_accuracy

# Example usage
predicted_texts = ["hello", "world", "foo"]
ground_truth_texts = ["helo", "world", "fooo"]

wra, character_accuracy = evaluate_metrics(predicted_texts, ground_truth_texts)

print(f"Word Recognition Accuracy (WRA): {wra:.4f}")
print(f"Character-level Accuracy (1 - NED): {character_accuracy:.4f}")
