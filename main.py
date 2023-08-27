from collections import Counter

import numpy as np
from gumbel_ranker import select_items_gumbel
from iterative_ranker import select_items
from mmr_ranker import select_items_with_embeddings

# Using iterative ranke
item_ids = [1, 2, 3, 4, 5]
item_relevance_scores = [0.8, 0.6, 0.9, 0.5, 0.7]
print(f"Ranking items \n\t{item_ids} with scores: \n\t{item_relevance_scores}")

selected_items = select_items(item_ids, item_relevance_scores)
print("Selected items:", selected_items)

# Using Gumbel ranker
# Perform the process a hundred times
num_trials = 1000
first_index_counts = Counter()

for _ in range(num_trials):
  selected_items_gumbel = select_items_gumbel(item_ids, item_relevance_scores)
  first_index_counts[selected_items_gumbel[0]] += 1

total_trials = num_trials
percentages = {item_id: count / total_trials * 100 for item_id, count in first_index_counts.items()}

# Print percentages in descending order
print("Percentages of items at first index (descending order):")
for item_id, percentage in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
    print(f"Item {item_id}: {percentage:.2f}%")

# Using MMR
user_embedding = np.array([0.2, 0.4, 0.6])
item_embeddings = np.array([
  [0.1, 0.2, 0.3], 
  [0.4, 0.5, 0.6], 
  [0.7, 0.8, 0.9],                          
  [0.2, 0.3, 0.4], 
  [0.5, 0.6, 0.7]])

# Using Gumbel ranker
# Perform the process a hundred times
num_trials = 1000
mmr_first_index_counts = Counter()

for _ in range(num_trials):
  selected_items_gumbel = select_items_with_embeddings(
    item_ids, 
    item_relevance_scores,
    user_embedding, 
    item_embeddings
  )
  mmr_first_index_counts[selected_items_gumbel[0]] += 1

total_trials = num_trials
percentages = {item_id: count / total_trials * 100 for item_id, count in mmr_first_index_counts.items()}

# Print percentages in descending order
print("Percentages of items at first index (descending order):")
for item_id, percentage in sorted(percentages.items(), key=lambda x: x[1], reverse=True):
    print(f"Item {item_id}: {percentage:.2f}%")
