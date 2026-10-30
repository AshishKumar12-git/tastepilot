import streamlit as st

from agent_controller import run_recommendation_agent


st.set_page_config(
    page_title="TastePilot | Qloo-powered recommendations",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


def safe_list(value):
    return value if isinstance(value, list) else []


def render_recommendations(recommendations):
    if not recommendations:
        st.info(
            "I couldn't find suitable options just yet. Try changing "
            "your preferences or search area, and I'll take another look."
        )
        return

    st.subheader("Your recommendations")

    for index, item in enumerate(recommendations, start=1):
        name = item.get("name") or "Unnamed recommendation"
        with st.container(border=True):
            st.markdown(f"#### {index}. {name}")


# Sidebar
with st.sidebar:
    st.title("✦ TastePilot")
    st.caption("Culturally grounded recommendations, powered by Qloo.")
    st.divider()
    st.markdown("### What makes it different?")
    st.write(
        "Your request is translated into structured intent, mapped to "
        "Qloo's taste graph, ranked using matching signals, and checked "
        "by an evaluator before recommendations are returned."
    )
    st.divider()
    st.caption(
        "Recommendations are generated from current Qloo retrieval. "
        "Shared memory helps the agent reuse prior execution lessons; "
        "it does not replace fresh retrieval."
    )

st.title("Discover what you'll love.")
st.write(
    "Tell me what you're looking for. Include a category, location, "
    "preferences, companions, or budget if they matter."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "pending_location_intent" not in st.session_state:
    st.session_state.pending_location_intent = None

# Render previous messages for this browser session.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("result"):
            render_recommendations(
                safe_list(message["result"].get("recommendations"))
            )

prompt = st.chat_input(
    "e.g. Find fine-dining restaurants in Bengaluru with a quiet ambience"
)

if prompt:
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            pending_intent = st.session_state.pending_location_intent
            is_location_reply = pending_intent is not None

            if is_location_reply and prompt.strip().lower() in {"cancel", "never mind", "nevermind"}:
                st.session_state.pending_location_intent = None
                message = "No problem. I cancelled that recommendation request. You can start a new one anytime."
                st.info(message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": message,
                })
            else:
                with st.spinner(
                    "Adding your city..." if is_location_reply else
                    "Understanding your request, consulting Qloo, "
                    "and evaluating recommendations..."
                ):
                    if is_location_reply:
                        result = run_recommendation_agent(
                            user_prompt=pending_intent.get("original_prompt", ""),
                            intent_override=pending_intent["intent"],
                            location_override=prompt.strip(),
                        )
                    else:
                        result = run_recommendation_agent(prompt)

                if result.get("status") == "needs_location":
                    st.session_state.pending_location_intent = {
                        "intent": result["intent"],
                        "original_prompt": prompt,
                    }
                    message = "Which city should I search in? Please enter the city name (or type **cancel** to stop)."
                    st.info(message)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": message,
                    })
                else:
                    # The pending request has now resumed; don't reuse its
                    # intent for any later, unrelated message.
                    if is_location_reply:
                        st.session_state.pending_location_intent = None

                    status = result.get("status")
                    if status == "blocked":
                        st.warning(result.get("message", "This request could not be processed."))
                        errors = safe_list(result.get("blocking_errors"))
                        if errors:
                            st.write("Mapping issues:", errors)
                    elif status == "no_results":
                        st.info(result.get("message", "No suitable results were found."))
                    else:
                        st.markdown(
                            result.get(
                                "message",
                                "Here are the recommendations I found.",
                            )
                        )

                    render_recommendations(
                        safe_list(result.get("recommendations"))
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.get(
                            "message",
                            "Here are the recommendations I found.",
                        ),
                        "result": result,
                    })

        except Exception as exc:
            st.error(
                "Something went wrong while generating recommendations. "
                "Check the application terminal for the error details."
            )
            st.exception(exc)
            # If a resumed request fails, retain the pending intent so the
            # user can retry with a city instead of losing the original ask.
            st.session_state.messages.append({
                "role": "assistant",
                "content": "The recommendation request failed. Please try again.",
            })
