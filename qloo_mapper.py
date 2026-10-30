
CATEGORY_ENTITY_TYPES = {
    "place": "urn:entity:place",
    "movie": "urn:entity:movie",
    "restaurant": "urn:entity:place",
}

CATEGORY_TAG_TYPE = "urn:tag:category:place"


# ---------------------------------------------------------
# General helpers
# ---------------------------------------------------------

def unique_values(values):
    """Remove duplicates while preserving order."""
    return list(dict.fromkeys(values))


def format_tag_ids(tag_ids):
    """
    Qloo GET requests expect comma-separated tag IDs.
    """
    return ",".join(unique_values(tag_ids))


# ---------------------------------------------------------
# Extract resolved tags
# ---------------------------------------------------------

def get_resolved_category_requirements(resolved_tags):
    """
    Flatten category requirements resolved by tag_resolver.py.

    These tags represent explicit category requirements
    and will be used as hard filters.
    """

    category_result = resolved_tags.get("category_result", {})

    category_requirements = []

    for item in category_result.get("resolved", []):
        concept = item.get("concept")

        for tag in item.get("tags", []):
            if tag.get("id"):
                category_requirements.append({
                    "concept": concept,
                    "tag_id": tag["id"],
                    "tag_type": tag.get("type"),
                })

    return category_requirements


def get_resolved_preferences(resolved_tags):
    """
    Flatten resolved preferences from tag_resolver.py.

    These tags are soft recommendation signals.
    """

    preferences = []

    for item in resolved_tags.get("resolved_preferences", []):
        concept = item.get("concept")
        family = item.get("family")
        source = item.get("source", "explicit")
        confidence = item.get("confidence", 1.0)

        for tag in item.get("tags", []):
            if tag.get("id"):
                preferences.append({
                    "concept": concept,
                    "family": family,
                    "tag_id": tag["id"],
                    "tag_type": tag.get("type"),
                    "source": source,
                    "confidence": confidence,
                })

    return preferences


# ---------------------------------------------------------
# Build common Insights parameters
# ---------------------------------------------------------

def build_base_params(intent, category):
    """
    Build common Qloo Insights API parameters.
    """

    params = {
        "filter.type": CATEGORY_ENTITY_TYPES[category],
        "take": intent.get("limit", 10),
        "feature.explainability": "true",
    }

    location = intent.get("location") or {}
    city = location.get("city")

    if city:
        params["filter.location.query"] = city

    return params


# ---------------------------------------------------------
# Build category filters
# ---------------------------------------------------------

def add_category_filters(params, category_requirements):
    """
    Add explicit category requirements as hard filters.

    Multiple requirements are combined using UNION (OR).

    Example:
        garden OR park

    This function only uses tags that were resolved and
    validated by tag_resolver.py.
    """

    category_tag_ids = [
        item["tag_id"]
        for item in category_requirements
        if item.get("tag_type") == CATEGORY_TAG_TYPE
    ]

    if category_tag_ids:
        params["filter.tags"] = format_tag_ids(category_tag_ids)

        # Explicitly request OR semantics.
        params["operator.filter.tags"] = "union"

    return params


# ---------------------------------------------------------
# Build soft preference signals
# ---------------------------------------------------------

def add_preference_signals(params, preferences):
    """
    Add resolved preferences as soft recommendation signals.

    These influence affinity/ranking rather than acting
    as hard category filters.
    """

    preference_tag_ids = [
        item["tag_id"]
        for item in preferences
        if item.get("tag_id")
    ]

    if preference_tag_ids:
        params["signal.interests.tags"] = format_tag_ids(
            preference_tag_ids
        )

    return params


# ---------------------------------------------------------
# Main mapper
# ---------------------------------------------------------

def map_intent_to_qloo_params(routed_request, resolved_tags):
    """
    Convert structured intent and resolved Qloo tags
    into Insights API parameters.

    Explicit category requirements:
        Hard filters through filter.tags.

    User preferences:
        Soft signals through signal.interests.tags.

    Unresolved explicit category requirements:
        Block the query to prevent silently broadening
        the user's request.
    """

    category = routed_request["category"]
    intent = routed_request["intent"]

    if category not in CATEGORY_ENTITY_TYPES:
        raise ValueError(
            f"Unsupported category: {category}"
        )

    if resolved_tags.get("category") != category:
        raise ValueError(
            "Resolved tags category does not match routed category."
        )

    # 1. Extract category requirements and preferences.
    category_requirements = get_resolved_category_requirements(
        resolved_tags
    )

    preferences = get_resolved_preferences(
        resolved_tags
    )

    category_result = resolved_tags.get(
        "category_result", {}
    )

    category_status = category_result.get(
        "status", "not_required"
    )

    unresolved_category_requirements = category_result.get(
        "unresolved", []
    )

    # 2. Do not silently discard unresolved explicit requirements.
    if category_status in ("unresolved", "partially_resolved"):

        return {
            "category": category,
            "strategy": routed_request.get("strategy"),
            "ready_for_query": False,
            "blocking_errors": [
                {
                    "type": "unresolved_category_requirement",
                    "details": unresolved_category_requirements,
                }
            ],
            "params": None,
            "resolved_category_requirements": category_requirements,
            "resolved_preferences": preferences,
            "unresolved_category_requirements": (
                unresolved_category_requirements
            ),
            "unresolved_preferences": resolved_tags.get(
                "unresolved_preferences", []
            ),
            "context": intent.get("context", {}),
            "constraints": intent.get("constraints", {}),
            "reference_entities": intent.get(
                "reference_entities", []
            ),
            "time_expression": intent.get("time_expression"),
        }

    # 3. Build common Insights parameters.
    params = build_base_params(
        intent=intent,
        category=category,
    )

    # 4. Add explicit category filters.
    add_category_filters(
        params=params,
        category_requirements=category_requirements,
    )

    # 5. Add soft preference signals.
    add_preference_signals(
        params=params,
        preferences=preferences,
    )

    # 6. Return the complete mapping result.
    return {
        "category": category,
        "strategy": routed_request.get("strategy"),
        "ready_for_query": True,
        "blocking_errors": [],
        "params": params,
        "resolved_category_requirements": category_requirements,
        "resolved_preferences": preferences,
        "unresolved_category_requirements": (
            unresolved_category_requirements
        ),
        "unresolved_preferences": resolved_tags.get(
            "unresolved_preferences", []
        ),
        "context": intent.get("context", {}),
        "constraints": intent.get("constraints", {}),
        "reference_entities": intent.get(
            "reference_entities", []
        ),
        "time_expression": intent.get("time_expression"),
    }
