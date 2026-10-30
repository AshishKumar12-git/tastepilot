
import re


# ---------------------------------------------------------
# Ranking configuration
# ---------------------------------------------------------

EXPLICIT_WEIGHT = 1.0
INFERRED_WEIGHT = 0.5

MIN_INFERRED_CONFIDENCE = 0.70

MAX_SCORE = 100

CATEGORY_SCORE_WEIGHT = 0.70
PREFERENCE_SCORE_WEIGHT = 0.30


# ---------------------------------------------------------
# General helpers
# ---------------------------------------------------------

def normalize_text(value):
    """Normalize text for matching and deduplication."""

    if not isinstance(value, str):
        return ""

    value = value.lower().strip().replace("_", " ")

    # Remove punctuation so Glen's and Glens normalize similarly.
    value = re.sub(r"[^a-z0-9\s]", "", value)

    return " ".join(value.split())


# ---------------------------------------------------------
# Extract Qloo entities
# ---------------------------------------------------------

def extract_candidate_entities(qloo_response):

    if not isinstance(qloo_response, dict):
        raise ValueError(
            "Qloo response must be a dictionary."
        )

    results = qloo_response.get("results", {})

    if not isinstance(results, dict):
        return []

    entities = results.get("entities", [])

    return entities if isinstance(entities, list) else []


# ---------------------------------------------------------
# Extract entity tags
# ---------------------------------------------------------

def extract_candidate_tags(candidate):

    tags = candidate.get("tags", [])

    if not isinstance(tags, list):
        return []

    extracted = []

    for tag in tags:

        if isinstance(tag, dict):

            tag_id = tag.get("id")
            name = tag.get("name", "")
            tag_type = tag.get("type")

            if tag_id or name:
                extracted.append({
                    "id": tag_id,
                    "name": name,
                    "type": tag_type,
                })

        elif isinstance(tag, str):

            extracted.append({
                "id": tag,
                "name": "",
                "type": None,
            })

    return extracted


def extract_primary_genre(candidate):
    """Extract Qloo's primary genre/category when available."""

    properties = candidate.get("properties", {})
    if not isinstance(properties, dict):
        return None

    primary = properties.get("primary_genre")
    if not isinstance(primary, dict):
        return None

    tag_id = primary.get("id")
    name = primary.get("name")

    if not tag_id and not name:
        return None

    return {
        "id": tag_id,
        "name": name or "",
        "normalized_name": normalize_text(name or ""),
        "type": primary.get("type"),
    }


def primary_genre_matches_requirements(primary_genre, requirements):
    """Return whether the primary genre matches any requested category."""

    if not primary_genre or not requirements:
        return None

    for requirement in requirements:
        id_match = (
            primary_genre.get("id")
            and primary_genre["id"] == requirement["tag_id"]
        )
        name_match = (
            primary_genre.get("normalized_name")
            and primary_genre["normalized_name"]
            == requirement["normalized_name"]
        )

        if id_match or name_match:
            return True

    return False


def extract_matched_preference_tags(candidate_tags, preferences):
    """Return the actual candidate tags supporting matched preferences."""

    candidate_matches = []

    for tag in candidate_tags:
        tag_id = tag.get("id")
        tag_name = normalize_text(tag.get("name", ""))

        for preference in preferences:
            if (
                (tag_id and tag_id in preference["tag_ids"])
                or (tag_name and tag_name in preference["tag_names"])
            ):
                candidate_matches.append({
                    "name": tag.get("name"),
                    "id": tag_id,
                    "preference": preference["concept"],
                })
                break

    return candidate_matches


# ---------------------------------------------------------
# Extract candidate identity
# ---------------------------------------------------------

def extract_candidate_identity(candidate):

    candidate_id = (
        candidate.get("entity_id")
        or candidate.get("id")
    )

    name = (
        candidate.get("name")
        or candidate.get("title")
        or candidate.get("name_display")
        or "Unknown"
    )

    return candidate_id, name


# ---------------------------------------------------------
# Extract requested category requirements
# ---------------------------------------------------------

def extract_category_requirements(tag_resolution):
    """
    Read category requirements resolved by tag_resolver.py.

    Example:
        bakery
        dessert shop

    Each requirement contains its Qloo tag ID and name.
    """

    category_result = tag_resolution.get(
        "category_result", {}
    )

    requirements = []

    for item in category_result.get("resolved", []):

        concept = item.get("concept", "")

        for tag in item.get("tags", []):

            if not isinstance(tag, dict):
                continue

            if not tag.get("id"):
                continue

            requirements.append({
                "concept": concept,
                "tag_id": tag["id"],
                "tag_name": tag.get("name", concept),
                "normalized_name": normalize_text(
                    tag.get("name", concept)
                ),
            })

    return requirements


# ---------------------------------------------------------
# Category matching
# ---------------------------------------------------------

def match_category_requirements(
    candidate_tags,
    category_requirements,
):
    """
    Check whether a candidate matches the requested categories.

    A candidate qualifies if it matches at least one requested
    category requirement.

    Matching uses:
        1. Exact Qloo tag ID.
        2. Exact normalized Qloo tag name.

    Unrelated tags never contribute to category matching.
    """

    matched = []

    candidate_tag_ids = {
        tag["id"]
        for tag in candidate_tags
        if tag.get("id")
    }

    candidate_tag_names = {
        normalize_text(tag.get("name", ""))
        for tag in candidate_tags
        if tag.get("name")
    }

    for requirement in category_requirements:

        id_match = (
            requirement["tag_id"] in candidate_tag_ids
        )

        name_match = (
            requirement["normalized_name"]
            in candidate_tag_names
        )

        if id_match or name_match:

            matched.append({
                "concept": requirement["concept"],
                "tag_id": requirement["tag_id"],
                "tag_name": requirement["tag_name"],
                "match_method": (
                    "tag_id" if id_match else "tag_name"
                ),
            })

    return matched


# ---------------------------------------------------------
# Extract resolved preferences
# ---------------------------------------------------------

def extract_resolved_preferences(tag_resolution):

    resolved_preferences = tag_resolution.get(
        "resolved_preferences", []
    )

    preferences = []

    for item in resolved_preferences:

        concept = item.get("concept")

        if not concept:
            continue

        source = item.get("source", "explicit")
        confidence = item.get("confidence", 1.0)

        if source == "inferred":

            if confidence < MIN_INFERRED_CONFIDENCE:
                continue

            weight = INFERRED_WEIGHT * confidence

        else:
            weight = EXPLICIT_WEIGHT

        tag_ids = set()
        tag_names = set()

        for tag in item.get("tags", []):

            if not isinstance(tag, dict):
                continue

            if tag.get("id"):
                tag_ids.add(tag["id"])

            if tag.get("name"):
                tag_names.add(
                    normalize_text(tag["name"])
                )

        if not tag_ids and not tag_names:
            continue

        preferences.append({
            "concept": concept,
            "source": source,
            "weight": weight,
            "tag_ids": tag_ids,
            "tag_names": tag_names,
        })

    return preferences


# ---------------------------------------------------------
# Preference matching
# ---------------------------------------------------------

def calculate_preference_score(
    candidate_tags,
    preferences,
):
    """
    Score only successfully resolved user preferences.

    Parking, accessibility, and other unrelated metadata
    are not included unless explicitly requested.
    """

    candidate_tag_ids = {
        tag["id"]
        for tag in candidate_tags
        if tag.get("id")
    }

    candidate_tag_names = {
        normalize_text(tag.get("name", ""))
        for tag in candidate_tags
        if tag.get("name")
    }

    total_weight = sum(
        preference["weight"]
        for preference in preferences
    )

    matched_weight = 0.0

    matched_preferences = []
    unmatched_preferences = []

    for preference in preferences:

        id_match = bool(
            preference["tag_ids"] & candidate_tag_ids
        )

        name_match = bool(
            preference["tag_names"] & candidate_tag_names
        )

        if id_match or name_match:

            matched_weight += preference["weight"]

            matched_preferences.append(
                preference["concept"]
            )

        else:

            unmatched_preferences.append(
                preference["concept"]
            )

    score = (
        matched_weight / total_weight * MAX_SCORE
        if total_weight > 0
        else 0.0
    )

    return {
        "score": round(score, 2),
        "matched_preferences": matched_preferences,
        "unmatched_preferences": unmatched_preferences,
        "matched_preference_tags": extract_matched_preference_tags(
            candidate_tags,
            preferences,
        ),
    }


# ---------------------------------------------------------
# Deduplication
# ---------------------------------------------------------

def deduplicate_candidates(candidates):
    """
    Remove duplicate entities.

    First deduplicate by entity ID.
    Then deduplicate by normalized business name.

    This intentionally collapses repeated branches with
    the same displayed business name for this initial test.
    """

    unique = []
    seen_ids = set()
    seen_names = set()

    for candidate in candidates:

        candidate_id, name = extract_candidate_identity(
            candidate
        )

        normalized_name = normalize_text(name)

        if candidate_id and candidate_id in seen_ids:
            continue

        if normalized_name and normalized_name in seen_names:
            continue

        if candidate_id:
            seen_ids.add(candidate_id)

        if normalized_name:
            seen_names.add(normalized_name)

        unique.append(candidate)

    return unique


# ---------------------------------------------------------
# Calculate one candidate's score
# ---------------------------------------------------------

def calculate_candidate_score(
    candidate,
    category_requirements,
    preferences,
):
    candidate_tags = extract_candidate_tags(candidate)
    primary_genre = extract_primary_genre(candidate)

    # Match requested categories using the existing tag-level logic.
    matched_categories = match_category_requirements(
        candidate_tags,
        category_requirements,
    )

    category_score = (
        len(matched_categories)
        / len(category_requirements)
        * MAX_SCORE
        if category_requirements
        else 0.0
    )

    # Match user preferences.
    preference_result = calculate_preference_score(
        candidate_tags,
        preferences,
    )

    preference_score = preference_result["score"]

    # Calculate final score.
    if category_requirements and preferences:

        final_score = (
            category_score * CATEGORY_SCORE_WEIGHT
            + preference_score * PREFERENCE_SCORE_WEIGHT
        )

    elif category_requirements:

        final_score = category_score

    else:

        final_score = preference_score

    candidate_id, name = extract_candidate_identity(
        candidate
    )

    return {
        "id": candidate_id,
        "name": name,
        "score": round(final_score, 2),

        "score_breakdown": {
            "category_match": round(category_score, 2),
            "preference_match": preference_score,
            "matched_categories": [
                item["concept"]
                for item in matched_categories
            ],
            "matched_preferences": preference_result[
                "matched_preferences"
            ],
            "unmatched_preferences": preference_result[
                "unmatched_preferences"
            ],
            "matched_preference_tags": preference_result[
                "matched_preference_tags"
            ],
            "primary_genre": (
                {
                    "id": primary_genre["id"],
                    "name": primary_genre["name"],
                }
                if primary_genre
                else None
            ),
        },

        "relevant_tags": (
            [
                {
                    "name": tag.get("name"),
                    "id": tag.get("id"),
                    "match_type": "category",
                }
                for tag in candidate_tags
                if any(
                    tag.get("id") == requirement["tag_id"]
                    or normalize_text(tag.get("name", ""))
                    == requirement["normalized_name"]
                    for requirement in category_requirements
                )
            ]
            + [
                {
                    "name": tag.get("name"),
                    "id": tag.get("id"),
                    "match_type": "preference",
                    "preference": tag.get("preference"),
                }
                for tag in preference_result["matched_preference_tags"]
            ]
        ),

        "original_data": candidate,
    }


# ---------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------


# ---------------------------------------------------------
# Main ranking function
# ---------------------------------------------------------

def rank_recommendations(
    entities_or_intent,
    tag_resolution,
    qloo_response=None
):
    """
    Validate, deduplicate, and rank Qloo recommendations.

    Supports both calling styles:

    1. rank_recommendations(entities, tag_resolution)

    2. rank_recommendations(intent, tag_resolution, qloo_response)

    Category requirements:
        Hard inclusion criteria.

    Preferences:
        Soft ranking signals.
    """

    if not isinstance(tag_resolution, dict):
        raise TypeError(
            "tag_resolution must be a dictionary."
        )

    # -----------------------------------------------------
    # 1. Extract entities
    # -----------------------------------------------------

    if qloo_response is not None:

        # Three-argument calling style.
        entities = extract_candidate_entities(qloo_response)

    elif isinstance(entities_or_intent, list):

        # Two-argument calling style: entities already extracted.
        entities = entities_or_intent

    elif isinstance(entities_or_intent, dict):

        # Two-argument calling style: complete Qloo response.
        entities = extract_candidate_entities(
            entities_or_intent
        )

    else:
        raise TypeError(
            "Expected entities list or Qloo response dictionary."
        )

    # -----------------------------------------------------
    # 2. Extract requested categories and preferences
    # -----------------------------------------------------

    category_requirements = extract_category_requirements(
        tag_resolution
    )

    preferences = extract_resolved_preferences(
        tag_resolution
    )

    ranked = []

    # -----------------------------------------------------
    # 3. Category validation and scoring
    # -----------------------------------------------------

    for candidate in entities:

        if not isinstance(candidate, dict):
            continue

        candidate_tags = extract_candidate_tags(candidate)
        primary_genre = extract_primary_genre(candidate)

        matched_categories = match_category_requirements(
            candidate_tags,
            category_requirements
        )

        # If Qloo supplies a primary genre, it must match a requested
        # category. Secondary tags cannot override a conflicting primary.
        primary_match = primary_genre_matches_requirements(
            primary_genre,
            category_requirements
        )

        if category_requirements:
            if primary_match is False:
                continue

            # If primary genre is absent, use the existing tag fallback.
            if primary_match is None and not matched_categories:
                continue

        result = calculate_candidate_score(
            candidate=candidate,
            category_requirements=category_requirements,
            preferences=preferences
        )

        ranked.append(result)

    # -----------------------------------------------------
    # 4. Sort by relevance before deduplication
    # -----------------------------------------------------

    ranked.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    # -----------------------------------------------------
    # 5. Deduplicate, preserving the highest-scoring record
    # -----------------------------------------------------

    unique_results = []

    seen_ids = set()
    seen_names = set()

    for result in ranked:

        candidate_id = result.get("id")
        normalized_name = normalize_text(
            result.get("name", "")
        )

        if candidate_id and candidate_id in seen_ids:
            continue

        if normalized_name and normalized_name in seen_names:
            continue

        if candidate_id:
            seen_ids.add(candidate_id)

        if normalized_name:
            seen_names.add(normalized_name)

        unique_results.append(result)

    # -----------------------------------------------------
    # 6. Assign final ranks
    # -----------------------------------------------------

    for position, result in enumerate(
        unique_results,
        start=1
    ):
        result["rank"] = position

    return unique_results

