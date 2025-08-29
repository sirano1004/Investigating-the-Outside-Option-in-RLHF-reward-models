import numpy as np

# Define a function to determine preference
def determine_preference(row, rng=np.random.default_rng(42)):
    """
    Simulates a preference ranking between two responses and a 'neither' option
    using the Gumbel-Max trick.

    Args:
        row (dict): A dictionary-like object with 'first_response_score' and
                    'second_response_score' keys.
        rng (np.random.Generator): A NumPy random number generator for reproducibility.

    Returns:
        str: A string representing the preference order (e.g., '1 > 2 > neither').
    """
    # Add Gumbel-distributed noise to each score
    score1 = row['first_response_score'] + rng.gumbel(0, 1)
    score2 = row['second_response_score'] + rng.gumbel(0, 1)
    score0 = 0 + rng.gumbel(0, 1) # The 'neither' option has a base score of 0

    # Determine the ranking by comparing the scores
    scores = {'1': score1, '2': score2, 'neither': score0}
    sorted_items = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    
    # Construct the preference string
    preference_order = ' > '.join([item[0] for item in sorted_items])

    return preference_order