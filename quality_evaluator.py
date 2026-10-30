import json

from intent_parser import client, MODEL


ALLOWED_DECISIONS = {
    "ACCEPT",
    "RETRY_WITH_MORE_CANDIDATES",
    "STOP",
}


def evaluate_recommendations(
    original_prompt,
    intent,
    ranked_results,
    retrieved_count,
    attempt,
    max_attempts,
    relevant_memories=None,
):
    compact_results = []
    for item in ranked_results[:8]:
        breakdown = item.get("score_breakdown", {})
        compact_results.append({
            "name": item.get("name"),
            "score": item.get("score"),
            "matched_categories": breakdown.get("matched_categories", []),
            "matched_preferences": breakdown.get("matched_preferences", []),
        })

    payload = {
        "original_prompt": original_prompt,
        "intent": intent,
        "retrieved_count": retrieved_count,
        "ranked_count": len(ranked_results),
        "top_results": compact_results,
        "attempt": attempt,
        "max_attempts": max_attempts,
        "relevant_memories": relevant_memories or {},
    }

    system_prompt = """
You are a recommendation quality evaluator.

Use current result evidence as the primary source. Memories are supporting
context only:
- Verified knowledge may be used as reusable factual context.
- Past execution observations describe what happened previously; they are
  not guarantees and must not override current evidence.
- Never infer candidate facts from memory.
- Explicit category requirements are hard constraints.
- Preferences are relevance signals, not hard constraints.
- Never recommend relaxing or removing explicit requirements.

Return JSON:
{
  "decision": "ACCEPT" | "RETRY_WITH_MORE_CANDIDATES" | "STOP",
  "reason": "brief explanation",
  "quality_score": integer from 0 to 100
}

Choose ACCEPT when results are relevant and supported.
Choose RETRY_WITH_MORE_CANDIDATES when more retrieval may help.
Choose STOP when evidence is inadequate or another retrieval is unlikely
to help. Do not change the user's original intent.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=400,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )

    raw = response.choices[0].message.content
    try:
        evaluation = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        evaluation = {}

    decision = evaluation.get("decision")
    if decision not in ALLOWED_DECISIONS:
        decision = "STOP"

    try:
        quality_score = int(evaluation.get("quality_score", 0))
    except (TypeError, ValueError):
        quality_score = 0

    reason = evaluation.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        reason = "Evaluator returned no usable explanation."

    if attempt >= max_attempts and decision == "RETRY_WITH_MORE_CANDIDATES":
        decision = "STOP"

    return {
        "decision": decision,
        "reason": reason.strip(),
        "quality_score": max(0, min(100, quality_score)),
    }
