from typing import List

import numpy as np


def select_items_gumbel(item_ids: List[int],
                        item_relevance_scores: List[float]) -> List[int]:
  scores = np.array(item_relevance_scores)
  gumbel_noise = -np.log(
      -np.log(np.random.rand(len(scores))))  # Sample Gumbel noise

  # print(f"scores = {scores}")
  noisy_scores = scores + gumbel_noise
  # print(f"noisy scores = {noisy_scores}")
  selected_indices = np.argsort(noisy_scores)[::-1][
      -len(item_ids):]  # Select items with highest noisy scores
  # print(f"selected_indices = {selected_indices}")

  selected_items = [item_ids[i] for i in selected_indices
                    ]  # Get the corresponding item IDs
  return selected_items
