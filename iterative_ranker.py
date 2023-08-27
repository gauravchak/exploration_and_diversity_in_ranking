def select_items(item_ids, item_relevance_scores):
  selected_items = []  # List to store selected item IDs

  # Iterate K times to select K items
  for _i in range(len(item_ids)):
    max_score = float('-inf')  # Initialize max_score to negative infinity
    selected_item = None

    # Find the item with the highest score among the ones not selected yet
    for item_id, score in zip(item_ids, item_relevance_scores):
      if item_id not in selected_items and score > max_score:
        max_score = score
        selected_item = item_id

    if selected_item is not None:
      selected_items.append(selected_item)  # Mark the selected item as chosen

  return selected_items
