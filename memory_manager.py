"""
Memory manager: creates a compact signature and records execution outcomes.

It deliberately does not convert free-form LLM explanations into verified
knowledge. Verified semantic memories must be added with evidence.
"""

import json

from memory_store import initialize_memory, save_episode


def build_memory_signature(intent):
    category = intent.get("category", "")
    requirements = intent.get("category_requirements", []) or []
    preferences = intent.get("preferences", []) or []

    requirement_names = []
    for item in requirements:
        if isinstance(item, dict):
            concept = item.get("concept")
        else:
            concept = item
        if concept:
            requirement_names.append(str(concept).strip().lower())

    preference_names = []
    for item in preferences:
        if isinstance(item, dict):
            concept = item.get("concept")
        else:
            concept = item
        if concept:
            preference_names.append(str(concept).strip().lower())

    location = intent.get("location") or {}
    city = location.get("city") if isinstance(location, dict) else None

    parts = [
        f"category {category}",
        f"requirements {' '.join(requirement_names)}",
        f"preferences {' '.join(preference_names)}",
    ]
    if city:
        parts.append(f"city {str(city).lower()}")

    return " ".join(parts)


def record_agent_run(intent, attempts, recommendations, status, final_evaluation):
    """Save aggregate, privacy-minimized outcome metrics."""
    initialize_memory()

    category = intent.get("category") or "unknown"
    signature = build_memory_signature(intent)

    if attempts:
        first = attempts[0]
        initial_take = first.get("requested_take")
        retrieved_count = sum(item.get("retrieved_count", 0) for item in attempts)
        ranked_count = len(recommendations)
    else:
        initial_take = None
        retrieved_count = 0
        ranked_count = 0

    retry_improved = False
    if len(attempts) > 1:
        first_ranked = attempts[0].get("ranked_count", 0)
        later_best = max(
            (item.get("ranked_count", 0) for item in attempts[1:]),
            default=first_ranked,
        )
        retry_improved = later_best > first_ranked

    final_decision = (
        final_evaluation.get("decision")
        if isinstance(final_evaluation, dict)
        else None
    )

    save_episode(
        category=category,
        signature=signature,
        initial_take=initial_take,
        attempts=len(attempts),
        retrieved_count=retrieved_count,
        ranked_count=ranked_count,
        status=status,
        final_decision=final_decision,
        retry_improved=retry_improved,
    )
