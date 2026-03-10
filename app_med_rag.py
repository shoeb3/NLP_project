import streamlit as st
from rag_core import answer_with_rag, answer_with_rag_ayurveda


# Page config
st.set_page_config(
    page_title="Medical • Nutrition • Ayurveda Assistant",
    page_icon="🩺",
    layout="wide",
)

# Session state for chat history
if "history" not in st.session_state:
    # Each item: {"domain": "...", "question": "...", "answer": "..."}
    st.session_state["history"] = []

# Emergency keyword detector

EMERGENCY_KEYWORDS = [
    # General emergency / life-threatening
    "heart attack",
    "stroke",
    "unconscious",
    "not breathing",
    "can’t breathe",
    "cant breathe",
    "difficulty breathing",
    "severe bleeding",
    "bleeding heavily",
    "blood everywhere",
    "choking",
    "overdose",
    "poisoning",
    "seizure",
    "convulsions",
    "unresponsive",

    # Chest pain / cardiac
    "severe chest pain",
    "chest pain",
    "pain in my chest",
    "pressure in my chest",

    # Suicide / self harm
    "kill myself",
    "want to die",
    "don’t want to live",
    "dont want to live",
    "suicidal",
    "self harm",
    "self-harm",
    "cut myself",
]

def is_emergency(text: str) -> bool:
    """Very simple keyword-based emergency detector."""
    t = text.lower()
    return any(kw in t for kw in EMERGENCY_KEYWORDS)


# Custom CSS
st.markdown(
    """
    <style>
        .main-container {
            max-width: 900px;
            margin: 0 auto;
            padding-top: 1rem;
        }
        .chat-card {
            background-color: #ffffff;
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            box-shadow: 0 8px 20px rgba(0,0,0,0.06);
            border: 1px solid #e5e7eb;
        }
        .answer-box {
            margin-top: 0.5rem;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
        }
        .footer-text {
            font-size: 0.8rem;
            color: #6b7280;
            margin-top: 2rem;
            text-align: center;
        }
        .user-bubble {
            background-color: #eef2ff;
            padding: 0.5rem 0.75rem;
            border-radius: 12px;
            margin-bottom: 0.25rem;
            font-size: 0.9rem;
        }
        .assistant-bubble {
            background-color: #f3f4f6;
            padding: 0.5rem 0.75rem;
            border-radius: 12px;
            margin-bottom: 0.75rem;
            font-size: 0.9rem;
        }
        .turn-domain {
            font-size: 0.75rem;
            color: #6b7280;
            margin-bottom: 0.1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.markdown("### 🧭 Settings")
    domain = st.radio(
        "Knowledge domain",
        options=["Medical / Nutrition", "Ayurveda"],
        index=0,
    )

    clear_chat = st.button("🧹 Clear chat history")
    if clear_chat:
        st.session_state["history"] = []

    st.markdown("---")
    st.markdown("### 💡 Examples")
    if domain == "Medical / Nutrition":
        st.markdown(
            """
            - What diet is recommended for someone with diabetes?  
            - Can antibiotics cause skin rashes?  
            - How much water should an adult drink daily?
            """
        )
    else:
        st.markdown(
            """
            - According to Ayurveda, what supports good digestion?  
            - What is recommended for balancing Vata?  
            - Which Ayurvedic herbs are used for sleep?
            """
        )

    st.markdown("---")
    st.markdown(
        """
        **Disclaimer**  
        The content provided by this assistant is for informational and educational purposes only.
        It is not intended to be a substitute for professional medical advice, diagnosis, or treatment.
        Always seek the advice of your physician, dietitian, or other qualified health provider with any questions you may have regarding a medical condition..
        """
    )

# Main layout
st.markdown("<div class='main-container'>", unsafe_allow_html=True)

st.markdown("## 🩺 Medical • 🍎 Nutrition • 🌿 Ayurveda Assistant")
st.markdown(
    "Ask questions about modern medicine, nutrition, or traditional Ayurveda. "
    "Answers come from the curated knowledge base plus a language model."
)

st.markdown("")  # spacer

with st.container():
    st.markdown("<div class='chat-card'>", unsafe_allow_html=True)

    # --- Chat history display ---
    if st.session_state["history"]:
        st.markdown("### Conversation")
        for turn in st.session_state["history"]:
            st.markdown(
                f"<div class='turn-domain'>{turn['domain']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='user-bubble'><strong>You:</strong> {turn['question']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='assistant-bubble'><strong>Assistant:</strong> {turn['answer']}</div>",
                unsafe_allow_html=True,
            )
        st.markdown("---")
    else:
        st.markdown("### Conversation")
        st.caption("Ask your first question to start the conversation.")

    # --- New question input ---
    if domain == "Medical / Nutrition":
        label_text = "Medical / Nutrition mode – ask a question"
    else:
        label_text = "Ayurveda mode – ask a question"

    question = st.text_area(
        label_text,
        placeholder=(
            "Example (Medical / Nutrition): What diet is recommended for someone with diabetes?\n"
            "Example (Ayurveda): According to Ayurveda, what is good for digestion?"
        ),
        height=140,
        key="question_box",
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        ask_clicked = st.button("Ask", type="primary", use_container_width=True)
    with col2:
        st.caption("The model will answer briefly based on the retrieved context, unless this looks like an emergency.")

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question before submitting.")
        else:
            # --- Emergency detection first ---
            if is_emergency(question):
                emergency_msg = (
                    "🚨 **This may be an emergency.**\n\n"
                    "Your message sounds like it could describe a life-threatening or urgent medical situation.\n\n"
                    "- Call your local **emergency number (such as 911)** immediately, or\n"
                    "- Go to the **nearest emergency room** or contact a qualified doctor right away.\n\n"
                    "This assistant cannot handle emergencies or provide real-time medical help."
                )

                # Add this "answer" to history but do NOT call the LLM/RAG
                st.session_state["history"].append(
                    {
                        "domain": f"{domain} (EMERGENCY FLAGGED)",
                        "question": question.strip(),
                        "answer": emergency_msg,
                    }
                )

                # Show the warning in the UI
                st.error(emergency_msg)

                # Rerun so history updates cleanly
                st.rerun()

            # --- Normal non-emergency flow ---
            with st.spinner("Thinking..."):
                try:
                    if domain == "Medical / Nutrition":
                        answer = answer_with_rag(question)
                    else:
                        answer = answer_with_rag_ayurveda(question)

                    # Save this turn in history
                    st.session_state["history"].append(
                        {
                            "domain": domain,
                            "question": question.strip(),
                            "answer": answer.strip(),
                        }
                    )

                except Exception as e:
                    st.error(f"An error occurred:\n\n{e}")

            # Refresh so the new turn shows up in history at the top
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # close chat-card

st.markdown("---")

if domain == "Ayurveda":
    st.markdown(
        """
        ###  Important Ayurveda Disclaimer

        Ayurveda is a **traditional, holistic, long-term healing system**.  
        Its recommendations focus on **balance**, **lifestyle**, **digestion**,  
        and **gradual improvement** — not emergency response.

        - It is **not meant for urgent or severe medical conditions**  
        - It **should not replace** modern medical diagnosis or treatment  
        - If you experience **sudden pain, difficulty breathing, bleeding, loss of consciousness, chest pain, seizures, or any emergency symptoms**,  
          seek **immediate medical care (such as 911)**

        This assistant provides **general educational information only**,  
        not professional medical or Ayurvedic diagnosis.
        """
    )

st.markdown("</div>", unsafe_allow_html=True)  # close main-container
