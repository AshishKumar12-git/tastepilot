
import json
import os
import re

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"

api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    raise ValueError("NVIDIA_API_KEY is missing from .env")


client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key,
    timeout=45.0,
    max_retries=1,
)


# ============================================================
# 1. DEFAULT INTENT STRUCTURE
# ============================================================

def get_default_intent():

    return {
        "category": None,

        "category_requirements": [],

        "location": {
            "city": None,
            "country": None,
        },

        "preferences": [],

        "reference_entities": [],

        "context": {
            "audience": [],
            "occasion": None,
            "age_group": [],
            "dietary_requirements": [],
            "accessibility_requirements": [],
        },

        "time_expression": None,

        "constraints": {
            "budget": {
                "amount": None,
                "currency": None,
                "unit": None,
                "strictness": None,
            },
            "other": [],
        },

        "limit": 10,
    }


# ============================================================
# 2. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the intent extraction component of a context-aware
recommendation application.

Your responsibility is to understand what the user wants
and convert their natural-language request into structured JSON.

The application currently supports three top-level categories:

1. place
2. movie
3. restaurant

IMPORTANT ARCHITECTURAL PRINCIPLE:

You determine WHAT the user wants.

You do NOT determine Qloo tag IDs, Qloo tag types,
Qloo entity identifiers, or API parameters.

Those responsibilities belong to separate Python components.

Return exactly the following JSON structure:

{
    "category": "place | movie | restaurant",

    "category_requirements": [
        {
            "concept": "string",
            "source": "explicit"
        }
    ],

    "location": {
        "city": null,
        "country": null
    },

    "preferences": [
        {
            "concept": "string",
            "source": "explicit | inferred",
            "confidence": 1.0,
            "reason": null
        }
    ],

    "reference_entities": [
        {
            "name": "string",
            "entity_type": "string"
        }
    ],

    "context": {
        "audience": [],
        "occasion": null,
        "age_group": [],
        "dietary_requirements": [],
        "accessibility_requirements": []
    },

    "time_expression": null,

    "constraints": {
        "budget": {
            "amount": null,
            "currency": null,
            "unit": null,
            "strictness": null
        },
        "other": []
    },

    "limit": 10
}


============================================================
EXTRACTION RULES
============================================================

1. TOP-LEVEL CATEGORY

Identify the primary recommendation domain.

Use only:
- place
- movie
- restaurant

Examples:

- Restaurant, cafe, bakery, pub, restobar,
  food-serving establishment -> restaurant.

- Garden, park, museum, waterfall, tourist attraction,
  nature destination -> place.

- Movie, film, cinema discovery -> movie.

These examples explain the three broad domains.
They are NOT a predefined list of supported subcategories.

Do not create additional top-level categories.

Do not invent a category if the user's intent is genuinely unclear.


2. CATEGORY REQUIREMENTS

This is a very important field.

Extract the actual types or subtypes of entities
the user explicitly wants to discover.

Do not use a predefined category vocabulary.

Understand the user's wording dynamically.
IMPORTANT:
category_requirements must ALWAYS be a list of objects.

Each object must contain:
{
    "concept": "category name",
    "source": "explicit"
}

NEVER return category_requirements as a list of strings.

Example:
"category_requirements": [
    {
        "concept": "fine dining restaurant",
        "source": "explicit"
    }
]

Examples:

User: "Find restaurants in Bengaluru."

"category": "restaurant",
"category_requirements": [
    {
        "concept": "restaurant",
        "source": "explicit"
    }
]

User: "Find cafes and bakeries."

"category_requirements": [
    {
        "concept": "cafe",
        "source": "explicit"
    },
    {
        "concept": "bakery",
        "source": "explicit"
    }
]

User: "Find fine dining restaurants."

"category_requirements": [
    {
        "concept": "fine dining restaurant",
        "source": "explicit"
    }
]

User: "Find quiet gardens and parks."

"category_requirements": [
    {
        "concept": "garden",
        "source": "explicit"
    },
    {
        "concept": "park",
        "source": "explicit"
    }
]

User: "Find peaceful places to spend the weekend."

"category_requirements": []

User: "Recommend psychological thriller movies."

"category": "movie",
"category_requirements": [],
"preferences": [
    {
        "concept": "psychological thriller",
        "source": "explicit",
        "confidence": 1.0,
        "reason": null
    }
]

Rules:

- Extract category requirements only when explicitly requested.
- Preserve the user's intended entity type or subtype.
- Do not convert adjectives into category requirements.
- Do not convert preferences into hard categories.
- Do not invent categories based on the user's companions.
- When multiple acceptable entity types are explicitly requested,
  extract each separately.
- Multiple category requirements represent alternatives,
  not requirements that one entity must satisfy simultaneously.
- Do not create redundant broad and narrow requirements.
  For example, "fine dining restaurants" should normally
  produce "fine dining restaurant", not both "restaurant"
  and "fine dining restaurant".
- Never generate Qloo tag IDs or API-specific identifiers.


3. PREFERENCES

Extract characteristics that influence recommendation suitability.

Examples:

- quiet
- peaceful
- romantic
- nature
- adventurous
- family friendly
- fine dining atmosphere
- psychological thriller
- action
- Italian cuisine

Distinguish preferences from entity categories.

For example:

"Find quiet fine dining restaurants."

Category requirement:
- fine dining restaurant

Preference:
- quiet

Fine dining can also describe a dining experience.
Do not duplicate it as a separate preference unless
the user independently expresses that preference.

Explicit preferences:
- source = explicit
- confidence = 1.0
- reason = null unless an explanation is useful.

Inferred preferences:
- source = inferred
- confidence below 1.0
- include a short reason.

Inferred preferences must never become hard filters.

Do not invent preferences without contextual evidence.

Never infer medical or dietary requirements from age,
family relationships, or companion descriptions.


4. LOCATION

Extract city and country when mentioned.

Use null for unknown values.

Movies generally do not require a location unless
the user explicitly requests one.


5. REFERENCE ENTITIES

Extract named entities used as examples or references.

Examples:
- "I liked Inception."
- "Places similar to Lalbagh."
- "Restaurants similar to a named restaurant."

Identify their entity_type when possible.

Do not mistake ordinary preference concepts
for named entities.


6. CONTEXT

Extract:

- audience
- occasion
- age_group
- dietary_requirements
- accessibility_requirements

Do not invent requirements.

For example, parents accompanying the user does not
automatically imply vegetarian food or accessibility needs.

Never infer medical conditions.


7. TIME

Preserve expressions such as:

- this weekend
- tonight
- tomorrow
- next month

Do not invent dates or times.

Use null when unspecified.


8. CONSTRAINTS

Extract budget and other restrictions.

Use:

- hard: mandatory requirement.
- soft: preference or non-mandatory target.

Examples:

"Under 1000 INR" -> hard, unless context indicates otherwise.

"Preferably under 1000 INR" -> soft.

Never invent missing amounts or currencies.

Preserve other explicit restrictions.


9. RESULT LIMIT

Default to 10.

Respect an explicitly requested result count.


10. PRESERVATION RULE

Never silently drop an explicit requirement.

Preserve the meaning of the user's original wording.

If a concept is unfamiliar, extract the phrase
rather than replacing it with a guessed alternative.


Return valid JSON only.

Do not return Markdown, explanations, or additional text.
"""


# ============================================================
# 3. JSON EXTRACTION
# ============================================================

def parse_json_response(content):

    if not content or not isinstance(content, str):
        raise ValueError("Nemotron returned empty or invalid content.")

    content = content.strip()

    # Remove Markdown JSON fences.
    content = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        content,
        flags=re.IGNORECASE
    )
    content = re.sub(r"\s*```\s*$", "", content)

    try:
        intent = json.loads(content)

    except json.JSONDecodeError:
        # Fallback: extract the outermost JSON object.
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1 or end <= start:
            raise ValueError(
                f"No complete JSON object found. "
                f"Response preview: {content[:500]!r}"
            )

        try:
            intent = json.loads(content[start:end + 1])

        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON returned by Nemotron. "
                f"JSON error: {exc}. "
                f"Response preview: {content[:500]!r}"
            ) from exc

    if not isinstance(intent, dict):
        raise ValueError("Intent response must be a JSON object.")

    return intent


# ============================================================
# 4. DEFAULT NORMALIZATION
# ============================================================

def normalize_intent(intent):

    if not isinstance(intent, dict):
        raise ValueError("Intent must be a JSON object.")

    defaults = get_default_intent()

    # 1. Add missing top-level fields.
    for field, default_value in defaults.items():
        if field not in intent:
            intent[field] = default_value

    # 2. Normalize category requirements.
    requirements = intent["category_requirements"]

    if requirements is None:
        requirements = []

    if not isinstance(requirements, list):
        raise ValueError("category_requirements must be a list.")

    normalized_requirements = []

    for item in requirements:

        if isinstance(item, str):
            item = {
                "concept": item.strip(),
                "source": "explicit"
            }

        elif isinstance(item, dict):
            item = item.copy()
            item.setdefault("source", "explicit")

        else:
            raise ValueError(
                "Each category requirement must be a string or object."
            )

        normalized_requirements.append(item)

    intent["category_requirements"] = normalized_requirements

    # 3. Merge nested defaults.
    for field in ["location", "context", "constraints"]:

        if not isinstance(intent[field], dict):
            raise ValueError(f"{field} must be an object.")

        for key, value in defaults[field].items():
            intent[field].setdefault(key, value)

    # 4. Merge budget defaults.
    budget = intent["constraints"].get("budget")

    if budget is None:
        intent["constraints"]["budget"] = (
            defaults["constraints"]["budget"].copy()
        )

    elif isinstance(budget, dict):

        for key, value in defaults["constraints"]["budget"].items():
            budget.setdefault(key, value)

    else:
        raise ValueError(
            "constraints.budget must be an object or null."
        )

    return intent


# ============================================================
# 5. VALIDATION
# ============================================================

def validate_intent(intent):

    allowed_categories = {
        "place",
        "movie",
        "restaurant",
    }

    # ----------------------------------------
    # Top-level category
    # ----------------------------------------

    if intent["category"] not in allowed_categories:

        raise ValueError(
            f"Unsupported or missing category: "
            f"{intent['category']}"
        )

    # ----------------------------------------
    # Location
    # ----------------------------------------

    if not isinstance(intent["location"], dict):
        raise ValueError(
            "location must be an object."
        )

    # ----------------------------------------
    # Category requirements
    # ----------------------------------------

    if not isinstance(
        intent["category_requirements"], list
    ):
        raise ValueError(
            "category_requirements must be a list."
        )

    for requirement in intent["category_requirements"]:

        if not isinstance(requirement, dict):
            raise ValueError(
                "Each category requirement must be an object."
            )

        concept = requirement.get("concept")

        if not isinstance(concept, str) or not concept.strip():
            raise ValueError(
                "Category requirement concept must be non-empty."
            )

        if requirement.get("source") != "explicit":
            raise ValueError(
                "Only explicit category requirements are "
                "allowed to become hard category filters."
            )

    # ----------------------------------------
    # Preferences
    # ----------------------------------------

    if not isinstance(intent["preferences"], list):
        raise ValueError(
            "preferences must be a list."
        )

    for preference in intent["preferences"]:

        if not isinstance(preference, dict):
            raise ValueError(
                "Each preference must be an object."
            )

        concept = preference.get("concept")

        if not isinstance(concept, str) or not concept.strip():
            raise ValueError(
                "Preference concept must be non-empty."
            )

        if preference.get("source") not in {
            "explicit",
            "inferred",
        }:
            raise ValueError(
                "Invalid preference source."
            )

        confidence = preference.get("confidence")

        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= confidence <= 1
        ):
            raise ValueError(
                "Preference confidence must be between 0 and 1."
            )

        if (
            preference["source"] == "explicit"
            and confidence != 1.0
        ):
            raise ValueError(
                "Explicit preferences must have confidence 1.0."
            )

        if (
            preference["source"] == "inferred"
            and confidence >= 1.0
        ):
            raise ValueError(
                "Inferred preferences must have confidence below 1.0."
            )

    # ----------------------------------------
    # Reference entities
    # ----------------------------------------

    if not isinstance(
        intent["reference_entities"], list
    ):
        raise ValueError(
            "reference_entities must be a list."
        )

    for entity in intent["reference_entities"]:

        if not isinstance(entity, dict):
            raise ValueError(
                "Each reference entity must be an object."
            )

        if not isinstance(entity.get("name"), str):
            raise ValueError(
                "Reference entity name must be a string."
            )

    # ----------------------------------------
    # Context
    # ----------------------------------------

    if not isinstance(intent["context"], dict):
        raise ValueError(
            "context must be an object."
        )

    for field in [
        "audience",
        "age_group",
        "dietary_requirements",
        "accessibility_requirements",
    ]:

        if not isinstance(
            intent["context"].get(field), list
        ):
            raise ValueError(
                f"context.{field} must be a list."
            )

    # ----------------------------------------
    # Constraints
    # ----------------------------------------

    if not isinstance(intent["constraints"], dict):
        raise ValueError(
            "constraints must be an object."
        )

    if not isinstance(
        intent["constraints"].get("other"), list
    ):
        raise ValueError(
            "constraints.other must be a list."
        )

    # ----------------------------------------
    # Result limit
    # ----------------------------------------

    limit = intent["limit"]

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 1
    ):
        raise ValueError(
            "limit must be a positive integer."
        )

    return intent


# ============================================================
# 6. MAIN INTENT PARSER
# ============================================================

def parse_user_intent(user_prompt):

    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("User prompt must be a non-empty string.")

    max_attempts = 2
    last_error = None

    for attempt in range(1, max_attempts + 1):

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        # On retry, explicitly reinforce the output requirement.
        if attempt > 1:
            messages.append({
                "role": "assistant",
                "content": "The previous response did not produce valid intent JSON."
            })

            messages.append({
                "role": "user",
                "content": (
                    "Retry the original intent extraction. "
                    "Preserve every explicit requirement. "
                    "Return one complete JSON object matching the specified schema. "
                    "Do not include explanations or Markdown."
                )
            })

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=1800,

            # Request structured JSON output.
            response_format={"type": "json_object"},

            # Disable reasoning so tokens are available for JSON.
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        )

        choice = response.choices[0]
        content = choice.message.content
        finish_reason = choice.finish_reason

        print(f"\nIntent parser attempt: {attempt}")
        print(f"Finish reason: {finish_reason}")

        if not content:
            last_error = (
                f"Empty response. Finish reason: {finish_reason}"
            )

        elif finish_reason == "length":
            last_error = (
                "Response was truncated because the generation "
                "token limit was reached."
            )

        else:
            try:
                intent = parse_json_response(content)
                intent = normalize_intent(intent)
                intent = validate_intent(intent)

                return intent

            except (ValueError, TypeError, KeyError) as exc:
                last_error = str(exc)

        print(f"Intent extraction failed: {last_error}")

        if content:
            print(f"Response preview: {content[:500]!r}")

    raise ValueError(
        f"Intent extraction failed after {max_attempts} attempts. "
        f"Last error: {last_error}"
    )