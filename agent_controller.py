from copy import deepcopy

from intent_parser import parse_user_intent
from category_router import route_category
from tag_resolver import resolve_intent_tags
from qloo_mapper import map_intent_to_qloo_params
from qloo_client import get_insights
from ranker import rank_recommendations
from quality_evaluator import evaluate_recommendations
from response_generator import generate_response


MAX_ATTEMPTS = 3
MAX_TAKE = 50


def run_recommendation_agent(
    user_prompt,
    max_attempts=MAX_ATTEMPTS,
    intent_override=None,
    location_override=None,
):
    """
    Run the existing deterministic Qloo pipeline with a bounded
    evaluator-driven retrieval loop.

    The agent may increase retrieval breadth, but it never rewrites
    the parsed intent or removes explicit category requirements.
    """

    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError("user_prompt must be a non-empty string.")

    max_attempts = max(1, min(int(max_attempts), MAX_ATTEMPTS))

    # A location follow-up resumes the original request instead of parsing
    # the city name as a new recommendation prompt.
    if intent_override is not None:
        intent = deepcopy(intent_override)
    else:
        intent = parse_user_intent(user_prompt)

    if location_override is not None:
        location = intent.setdefault("location", {})
        if not isinstance(location, dict):
            location = {}
            intent["location"] = location
        location["city"] = str(location_override).strip()

    # Route first so we know whether this category needs a geographic scope.
    routed = route_category(intent)
    category = (routed.get("category") or "").strip().lower()
    location_dependent = category in {"place", "restaurant"}
    location = intent.get("location") or {}
    city = location.get("city") if isinstance(location, dict) else None

    if location_dependent and not (isinstance(city, str) and city.strip()):
        return {
            "status": "needs_location",
            "message": "Which city should I search in?",
            "intent": intent,
            "attempts": [],
            "recommendations": [],
        }

    tag_resolution = resolve_intent_tags(routed)

    mapping = map_intent_to_qloo_params(
        routed,
        tag_resolution,
    )

    if not mapping.get("ready_for_query"):
        return {
            "status": "blocked",
            "message": "The request could not be safely mapped to Qloo.",
            "blocking_errors": mapping.get("blocking_errors", []),
            "intent": intent,
            "tag_resolution": tag_resolution,
            "attempts": [],
            "recommendations": [],
        }

    base_params = deepcopy(mapping.get("params") or {})
    if not base_params:
        return {
            "status": "blocked",
            "message": "The mapper did not produce query parameters.",
            "blocking_errors": ["Missing Qloo query parameters."],
            "intent": intent,
            "tag_resolution": tag_resolution,
            "attempts": [],
            "recommendations": [],
        }

    try:
        original_take = int(base_params.get("take", intent.get("limit", 10)))
    except (TypeError, ValueError):
        original_take = 10

    original_take = max(1, original_take)
    attempt_history = []
    best_ranked = []
    final_evaluation = None

    for attempt_number in range(1, max_attempts + 1):
        params = deepcopy(base_params)

        # Broaden only the number of candidates retrieved. All category
        # filters and preference signals remain unchanged.
        params["take"] = min(
            original_take * (2 ** (attempt_number - 1)),
            MAX_TAKE,
        )

        qloo_response = get_insights(params)
        entities = (
            qloo_response.get("results", {}).get("entities", [])
            if isinstance(qloo_response, dict)
            else []
        )

        ranked = rank_recommendations(
            entities,
            tag_resolution,
        )

        evaluation = evaluate_recommendations(
            original_prompt=user_prompt,
            intent=intent,
            ranked_results=ranked,
            retrieved_count=len(entities),
            attempt=attempt_number,
            max_attempts=max_attempts,
        )

        attempt_history.append({
            "attempt": attempt_number,
            "requested_take": params["take"],
            "retrieved_count": len(entities),
            "ranked_count": len(ranked),
            "evaluation": evaluation,
        })

        if len(ranked) > len(best_ranked):
            best_ranked = ranked

        final_evaluation = evaluation

        if evaluation["decision"] == "ACCEPT":
            break

        if evaluation["decision"] == "STOP":
            break

        # RETRY is the only retry action the controller supports.
        # No other query parameters are modified.

    status = (
        "success"
        if best_ranked
        else "no_results"
    )

    user_facing_message = (
        generate_response(user_prompt, intent, best_ranked)
        if best_ranked
        else "I couldn't find suitable recommendations for that request. "
             "Try adjusting your preferences or search area."
    )

    return {
        "status": status,
        "message": user_facing_message,
        "intent": intent,
        "tag_resolution": tag_resolution,
        "attempts": attempt_history,
        "final_evaluation": final_evaluation,
        "recommendations": best_ranked,
    }


if __name__ == "__main__":
    prompt = input("What would you like recommendations for? ").strip()
    result = run_recommendation_agent(prompt)

    print("\nStatus:", result["status"])
    print("Message:", result["message"])

    if result.get("final_evaluation"):
        print("Evaluation:", result["final_evaluation"])

    print("\nAttempts:")
    for item in result["attempts"]:
        print(item)

    print("\nRecommendations:")
    for recommendation in result["recommendations"]:
        print(
            recommendation.get("rank"),
            recommendation.get("name"),
            recommendation.get("score"),
        )
