from difflib import SequenceMatcher

from qloo_client import search_qloo_tags


# Insights entity types and Qloo tag-search families are separate concepts.
CATEGORY_CONFIG = {
    "place": {
        "entity_type": "urn:entity:place",
        "category_tag_type": "urn:tag:category:place",
    },
    "restaurant": {
        "entity_type": "urn:entity:place",
        "category_tag_type": "urn:tag:category:place",
    },
    "movie": {
        "entity_type": "urn:entity:movie",
        "category_tag_type": None,
    },
}


# Preference concepts map to possible Qloo tag families.
# These are tag-family hypotheses, not hardcoded category/subcategory lists.
CONCEPT_TAG_TYPES = {
    "restaurant": {
        "cuisine": "urn:tag:cuisine:qloo",
        "ambience": "urn:tag:ambience:qloo",
        "dining_style": "urn:tag:culinary_style:qloo",
        "dining_option": "urn:tag:dining_option:qloo",
        "genre": "urn:tag:genre:place",
        "category": "urn:tag:category:place",
    },
    "place": {
        "ambience": "urn:tag:ambience:qloo",
        "setting": "urn:tag:setting:qloo",
        "genre": "urn:tag:genre:place",
        "category": "urn:tag:category:place",
    },
    "movie": {
        "genre": "urn:tag:genre:media",
    },
}


# ---------------------------------------------------------
# General helpers
# ---------------------------------------------------------

def normalize_text(value):
    if not isinstance(value, str):
        return ""

    return " ".join(
        value.lower().strip().replace("_", " ").split()
    )


def get_tag_parents(tag):
    parents = tag.get("parents", [])

    if not isinstance(parents, list):
        return []

    return [
        parent.get("type")
        for parent in parents
        if isinstance(parent, dict) and parent.get("type")
    ]


def extract_tags(response):
    if not isinstance(response, dict):
        return []

    results = response.get("results", {})

    if not isinstance(results, dict):
        return []

    tags = results.get("tags", [])

    return tags if isinstance(tags, list) else []


def lexical_similarity(query, candidate):
    query = normalize_text(query)
    candidate = normalize_text(candidate)

    if not query or not candidate:
        return 0.0

    if query == candidate:
        return 1.0

    if query in candidate or candidate in query:
        return 0.90

    return SequenceMatcher(None, query, candidate).ratio()


# ---------------------------------------------------------
# Qloo tag discovery and validation
# ---------------------------------------------------------

def search_tag_candidates(
    concept,
    category,
    tag_type,
    limit=20,
):
    """
    Discover tags from Qloo and validate their tag family
    and entity parent before considering them as candidates.
    """

    if not tag_type:
        return []

    response = search_qloo_tags(
        query=concept,
        limit=limit,
        extra_params={
            "filter.tag.types": tag_type,
        },
    )

    tags = extract_tags(response)

    candidates = []

    expected_parent = CATEGORY_CONFIG[category]["entity_type"]

    for tag in tags:
        tag_id = tag.get("id")
        actual_type = tag.get("type")
        parents = get_tag_parents(tag)

        if not tag_id:
            continue

        # Validate the returned tag family.
        if actual_type != tag_type:
            continue

        # Validate that the tag belongs to the expected entity type.
        if expected_parent not in parents:
            continue

        name = tag.get("name", "")

        candidates.append({
            "id": tag_id,
            "name": name,
            "type": actual_type,
            "parents": parents,
            "match_score": round(
                lexical_similarity(concept, name), 3
            ),
        })

    candidates.sort(
        key=lambda item: item["match_score"],
        reverse=True,
    )

    return candidates


def resolve_tag_concept(
    concept,
    category,
    tag_type,
    minimum_score=0.65,
):
    """
    Resolve a natural-language concept to a Qloo tag.

    A candidate is accepted only when its tag family,
    entity parent, and lexical similarity meet our checks.
    """

    candidates = search_tag_candidates(
        concept=concept,
        category=category,
        tag_type=tag_type,
    )

    if not candidates:
        return {
            "concept": concept,
            "status": "unresolved",
            "reason": "No matching Qloo tags found.",
            "tags": [],
        }

    best = candidates[0]

    if best["match_score"] < minimum_score:
        return {
            "concept": concept,
            "status": "unresolved",
            "reason": (
                "Best candidate has insufficient lexical similarity."
            ),
            "tags": candidates[:5],
        }

    return {
        "concept": concept,
        "status": "resolved",
        "tags": [best],
    }


# ---------------------------------------------------------
# Category requirement resolution
# ---------------------------------------------------------

def resolve_category_requirement(
    requirement,
    category,
):
    """
    Resolve one explicitly requested category or subtype.

    The concept comes dynamically from the LLM's
    category_requirements output.
    """

    concept = requirement.get("concept", "").strip()

    if not concept:
        return {
            "concept": concept,
            "status": "unresolved",
            "reason": "Empty category requirement.",
            "tags": [],
        }

    config = CATEGORY_CONFIG[category]
    tag_type = config["category_tag_type"]

    if not tag_type:
        return {
            "concept": concept,
            "status": "unsupported",
            "reason": (
                "No category tag family configured for this entity type."
            ),
            "tags": [],
        }

    return resolve_tag_concept(
        concept=concept,
        category=category,
        tag_type=tag_type,
    )


def resolve_category_tags(
    category,
    category_requirements,
):
    """
    Resolve all explicit category requirements.

    Multiple requirements are preserved separately.
    For example:
        garden
        park

    Neither requirement is silently discarded if unresolved.
    """

    resolved = []
    unresolved = []

    for requirement in category_requirements:
        result = resolve_category_requirement(
            requirement=requirement,
            category=category,
        )

        if result["status"] == "resolved":
            resolved.append(result)
        else:
            unresolved.append(result)

    if not category_requirements:
        status = "not_required"
    elif unresolved and resolved:
        status = "partially_resolved"
    elif unresolved:
        status = "unresolved"
    else:
        status = "resolved"

    return {
        "status": status,
        "resolved": resolved,
        "unresolved": unresolved,
    }


# ---------------------------------------------------------
# Preference family classification
# ---------------------------------------------------------

def infer_preference_family(concept, category):
    """
    Basic deterministic classification of preference concepts.

    This identifies which Qloo tag family to search.
    It does not determine whether a preference is relevant
    to the user's request.
    """

    text = normalize_text(concept)

    if category == "restaurant":

        if any(word in text for word in [
            "cuisine",
            "indian",
            "italian",
            "chinese",
            "japanese",
            "mexican",
            "south indian",
            "north indian",
            "vegetarian",
        ]):
            return "cuisine"

        if any(word in text for word in [
            "quiet",
            "romantic",
            "peaceful",
            "lively",
            "family friendly",
            "cozy",
            "ambience",
        ]):
            return "ambience"

        if any(word in text for word in [
            "fine dining",
            "casual dining",
            "casual",
        ]):
            return "dining_style"

        if any(word in text for word in [
            "pub",
            "bar",
            "nightlife",
            "restobar",
        ]):
            return "dining_option"

        return "category"

    if category == "place":

        if any(word in text for word in [
            "quiet",
            "peaceful",
            "tranquil",
            "lively",
        ]):
            return "ambience"

        if any(word in text for word in [
            "nature",
            "outdoor",
            "green",
        ]):
            return "setting"

        return "category"

    if category == "movie":
        return "genre"

    return None


# ---------------------------------------------------------
# Preference resolution
# ---------------------------------------------------------

def resolve_preference(
    preference,
    category,
):
    concept = preference.get("concept", "").strip()

    source = preference.get("source", "explicit")
    confidence = preference.get("confidence", 1.0)

    # Ignore low-confidence inferred preferences.
    if source == "inferred" and confidence < 0.70:
        return {
            "concept": concept,
            "status": "skipped",
            "reason": "Low-confidence inferred preference.",
            "tags": [],
        }

    family = infer_preference_family(
        concept=concept,
        category=category,
    )

    tag_type = CONCEPT_TAG_TYPES.get(
        category, {}
    ).get(family)

    if not tag_type:
        return {
            "concept": concept,
            "status": "unsupported",
            "reason": "No tag family configured for this preference.",
            "tags": [],
        }

    result = resolve_tag_concept(
        concept=concept,
        category=category,
        tag_type=tag_type,
    )

    result["family"] = family
    result["source"] = source
    result["confidence"] = confidence

    return result


# ---------------------------------------------------------
# Main intent tag resolution
# ---------------------------------------------------------

def resolve_intent_tags(routed_request):
    """
    Resolve category requirements and preferences separately.

    Category requirements:
        Explicit requirements that can become hard filters.

    Preferences:
        Soft recommendation signals used for ranking.
    """

    category = routed_request["category"]
    intent = routed_request["intent"]

    if category not in CATEGORY_CONFIG:
        raise ValueError(
            f"Unsupported category: {category}"
        )

    # 1. Resolve explicit category requirements dynamically.
    category_requirements = intent.get(
        "category_requirements", []
    )

    category_result = resolve_category_tags(
        category=category,
        category_requirements=category_requirements,
    )

    # 2. Resolve preferences independently.
    resolved_preferences = []
    unresolved_preferences = []

    for preference in intent.get("preferences", []):

        result = resolve_preference(
            preference=preference,
            category=category,
        )

        if result["status"] == "resolved":
            resolved_preferences.append(result)
        else:
            unresolved_preferences.append(result)

    # 3. Return structured results for the mapper and ranker.
    return {
        "category": category,

        "entity_type": CATEGORY_CONFIG[category]["entity_type"],

        "category_result": category_result,

        "resolved_preferences": resolved_preferences,

        "unresolved_preferences": unresolved_preferences,

        # Backward-compatible combined preference keys.
        "resolved_tags": resolved_preferences,

        "unresolved_tags": unresolved_preferences,
    }