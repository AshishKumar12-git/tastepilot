"""Lightweight retrieval from shared SQLite memory; no vector DB required."""

import json
import re

from memory_store import get_connection, initialize_memory


STOP_WORDS = {
    "the", "and", "with", "for", "from", "that", "this", "find",
    "show", "give", "some", "recommend", "recommendation", "in",
    "on", "to", "of", "a", "an", "is", "are", "me", "my",
}


def _tokens(value):
    words = re.findall(r"[a-z0-9]+", (value or "").lower())
    return {word for word in words if len(word) > 2 and word not in STOP_WORDS}


def _similarity(query, document):
    q = _tokens(query)
    d = _tokens(document)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def retrieve_memories(category, signature, limit=5):
    """
    Return relevant verified knowledge and similar past execution patterns.
    Episode matches are observations, not guaranteed rules.
    """
    initialize_memory()
    query = f"{category or ''} {signature or ''}"

    with get_connection() as conn:
        knowledge_rows = conn.execute("""
            SELECT * FROM semantic_memory
            WHERE verified = 1
              AND (category IS NULL OR category = ?)
            ORDER BY confidence DESC, updated_at DESC
        """, (category,)).fetchall()

        episode_rows = conn.execute("""
            SELECT * FROM episodic_memory
            WHERE category = ?
            ORDER BY created_at DESC
            LIMIT 200
        """, (category,)).fetchall()

    knowledge = []
    for row in knowledge_rows:
        doc = f"{row['title']} {row['content']} {row['tags']}"
        score = _similarity(query, doc)
        if score > 0 or not signature:
            knowledge.append({
                "type": "verified_knowledge",
                "title": row["title"],
                "content": row["content"],
                "confidence": row["confidence"],
                "source": row["source"],
                "similarity": round(score, 3),
            })

    episodes = []
    for row in episode_rows:
        score = _similarity(query, row["signature"])
        if score >= 0.12:
            episodes.append({
                "type": "past_execution_observation",
                "signature": row["signature"],
                "attempts": row["attempts"],
                "retrieved_count": row["retrieved_count"],
                "ranked_count": row["ranked_count"],
                "status": row["status"],
                "final_decision": row["final_decision"],
                "retry_improved": bool(row["retry_improved"]),
                "similarity": round(score, 3),
            })

    knowledge.sort(key=lambda item: (item["similarity"], item["confidence"]), reverse=True)
    episodes.sort(key=lambda item: item["similarity"], reverse=True)

    return {
        "verified_knowledge": knowledge[:limit],
        "past_experiences": episodes[:limit],
    }
