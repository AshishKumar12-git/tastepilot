import json

from intent_parser import client, MODEL


def _fallback_message(intent):
    location = intent.get("location") or {}
    city = location.get("city") if isinstance(location, dict) else None
    category = (intent.get("category") or "options").strip().lower()
    context = intent.get("context") or {}
    audience = context.get("audience") or []
    occasion = context.get("occasion")

    if category == "restaurant":
        subject = "restaurants"
    elif category == "place":
        subject = "places"
    else:
        subject = category if category.endswith("s") else f"{category}s"

    if city and audience:
        people = ", ".join(str(x) for x in audience)
        return (
            f"Planning a {subject[:-1] if subject.endswith('s') else subject} "
            f"outing in {city} with {people}? I've put together a shortlist "
            "to help you decide where to go."
        )
    if city and occasion:
        return (
            f"Got it — you're looking for {subject} in {city} for {occasion}. "
            "Here are a few options to consider."
        )
    if city:
        return (
            f"Looking for {subject} in {city}? Here’s a shortlist to help "
            "you find a spot that fits your plans."
        )
    return "I’ve put together a shortlist based on what you’re looking for."


def generate_response(user_prompt, intent, recommendations):
    """
    Generate a natural, intent-aware introduction for the recommendation list.
    The model can shape the language, but must not invent venue-level facts.
    """
    if not recommendations:
        return (
            "I couldn't find a shortlist that fits this search yet. "
            "Try changing the location or adjusting your preferences."
        )

    names = [
        item.get("name")
        for item in recommendations
        if isinstance(item, dict) and item.get("name")
    ]

    location = intent.get("location") or {}
    city = location.get("city") if isinstance(location, dict) else None
    category = intent.get("category") or "options"
    context = intent.get("context") or {}
    constraints = intent.get("constraints") or {}
    budget = constraints.get("budget") or {}

    payload = {
        "user_request": user_prompt,
        "category": category,
        "city": city,
        "preferences": intent.get("preferences") or [],
        "audience": context.get("audience") or [],
        "occasion": context.get("occasion"),
        "budget": budget,
        "recommendation_names": names[:8],
    }

    system_prompt = """
You are the warm, perceptive host of a high-quality discovery app.
Write the short introduction that appears immediately above a list of
recommendations.

Your writing should feel specific to THIS request, not like a generic
chatbot template:
- Start by reflecting the user's actual plan or intent in natural language.
- Make the transition into the shortlist feel confident and inviting.
- Use a little personality, but do not use exaggerated marketing language.
- Vary sentence structure. Avoid stock openings like "I've found some
  great spots", "Here are your recommendations", or "Looking for X?"
- Do not say the results are "the best", "top-rated", "popular", "nearby",
  "affordable", or "perfect" unless the supplied data proves that claim.
- Do not invent facts about any venue: no ambience, music, crowd, food,
  drinks, prices, ratings, opening hours, distance, or availability.
- Do not invent preferences or companions. Only reflect supplied context.
- Do not repeat venue names; the list is displayed separately.
- Do not mention Qloo, APIs, AI, agents, evaluation, scores, retries,
  ranking, or internal processing.
- Keep it to 1–2 concise sentences, ideally 20–35 words.
- If the prompt is very short, do not pad the response with filler.

Return only a JSON object in this shape:
{"message": "..."}

Example tone (do not copy verbatim):
User wants pubs in Bengaluru to hang out with friends.
"Making plans with your friends in Bengaluru? Start with this pub shortlist
and see which spot feels right for your next get-together."
"""

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.75,
            max_tokens=120,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            },
        )
        parsed = json.loads(response.choices[0].message.content)
        message = parsed.get("message")
        if isinstance(message, str) and message.strip():
            return message.strip()
    except Exception:
        # Preserve the recommendation flow if generation fails.
        pass

    return _fallback_message(intent)
