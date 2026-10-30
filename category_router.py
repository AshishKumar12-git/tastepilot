SUPPORTED_CATEGORIES = {
    "place": "place_strategy",
    "movie": "movie_strategy",
    "restaurant": "restaurant_strategy",
}


def route_category(intent):

    if not isinstance(intent, dict):
        raise TypeError("Intent must be a dictionary.")

    category = intent.get("category")

    if not isinstance(category, str):
        raise ValueError("Intent category must be a string.")

    category = category.strip().lower()

    strategy = SUPPORTED_CATEGORIES.get(category)

    if strategy is None:
        raise ValueError(
            f"Unsupported recommendation category: {category}"
        )

    return {
        "category": category,
        "strategy": strategy,
        "intent": intent
    }
