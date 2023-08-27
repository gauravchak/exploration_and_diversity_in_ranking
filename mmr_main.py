import numpy as np

from mmr_ranker import select_items_with_embeddings

# Example input
item_ids = [1, 2, 3, 4, 5]
item_relevance_scores = [0.8, 0.6, 0.9, 0.5, 0.7]
user_embedding = np.array([0.2, 0.4, 0.6])
item_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9],
                            [0.2, 0.3, 0.4], [0.5, 0.6, 0.7]])

selected_items = select_items_with_embeddings(item_ids, item_relevance_scores,
                                              user_embedding, item_embeddings)
print("Selected items with embeddings:", selected_items)
