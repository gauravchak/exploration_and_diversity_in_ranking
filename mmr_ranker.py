from typing import List

import numpy as np


def select_items_with_embeddings(item_ids: List[int],
                                 item_relevance_scores: List[float],
                                 user_embedding: np.ndarray,
                                 item_embeddings: np.ndarray,
                                 mmr_penalty_wt: float = 0.2,
                                 gumbel_wt: float = 1) -> List[int]:
  """
  Run MMR after adding Gumbel noise
  """

  # Use the Gumbel trick for Thompson Sampling

  gumbel_noise = -np.log(-np.log(np.random.rand(len(item_ids)))) * gumbel_wt
  marginal_relevance = item_relevance_scores + gumbel_noise

  selected_items = []

  while len(selected_items) < len(item_ids):
    next_item_idx = np.argmax(marginal_relevance)
    marginal_relevance[next_item_idx] = float('-inf')
    selected_item = item_ids[next_item_idx]
    selected_items.append(selected_item)

    # Remove the part of the user embedding that is explained by this item
    user_embedding_next = user_embedding - np.dot(
        user_embedding, item_embeddings[next_item_idx]) / np.dot(
            item_embeddings[next_item_idx],
            item_embeddings[next_item_idx]) * item_embeddings[next_item_idx]
    # remove the part of the user embedding that was catered to
    # by the just sleected item
    del_user = (user_embedding_next - user_embedding)
    for i, current_relevance in enumerate(marginal_relevance):
      if current_relevance > -100:
        # if only allows unselected items
        # from relevance deduct the part that correponds to
        # the interest already captured by items preselected in the list.
        marginal_relevance[i] += mmr_penalty_wt * np.dot(
            del_user, item_embeddings[i])
    user_embedding = user_embedding_next

  return selected_items
