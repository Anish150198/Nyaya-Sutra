"""
Widget-based demographic input form for the Welfare Agent (Sahayak).
Collects user profile and finds matching government schemes.
"""

import streamlit as st
from core.data_models import UserQuery, UserProfile, Persona, ActFilter, OrchestratorResponse
from app.state_manager import set_user_profile


INDIAN_STATES = [
    "AN", "AP", "AR", "AS", "BR", "CH", "CT", "DD", "DL", "GA",
    "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH",
    "ML", "MN", "MP", "MZ", "NL", "OD", "PB", "PY", "RJ", "SK",
    "TN", "TG", "TR", "UK", "UP", "WB",
]

STATE_NAMES = {
    "AN": "Andaman & Nicobar", "AP": "Andhra Pradesh", "AR": "Arunachal Pradesh",
    "AS": "Assam", "BR": "Bihar", "CH": "Chandigarh", "CT": "Chhattisgarh",
    "DD": "Daman & Diu", "DL": "Delhi", "GA": "Goa", "GJ": "Gujarat",
    "HP": "Himachal Pradesh", "HR": "Haryana", "JH": "Jharkhand",
    "JK": "Jammu & Kashmir", "KA": "Karnataka", "KL": "Kerala", "LA": "Ladakh",
    "LD": "Lakshadweep", "MH": "Maharashtra", "ML": "Meghalaya",
    "MN": "Manipur", "MP": "Madhya Pradesh", "MZ": "Mizoram",
    "NL": "Nagaland", "OD": "Odisha", "PB": "Punjab", "PY": "Puducherry",
    "RJ": "Rajasthan", "SK": "Sikkim", "TN": "Tamil Nadu", "TG": "Telangana",
    "TR": "Tripura", "UK": "Uttarakhand", "UP": "Uttar Pradesh", "WB": "West Bengal",
}


def render_scheme_wizard(handle_fn):
    """
    Render the Scheme Wizard form and results.

    Parameters
    ----------
    handle_fn : Callable[[UserQuery], OrchestratorResponse]
        The orchestrator.handle function.
    """
    st.subheader("🏛️ Government Scheme Finder")
    st.markdown(
        "Fill in your details below to find government schemes and benefits "
        "you may be eligible for."
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        gender = st.selectbox("Gender", ["male", "female", "other"])
        state_idx = st.selectbox(
            "State / UT",
            range(len(INDIAN_STATES)),
            format_func=lambda i: f"{INDIAN_STATES[i]} – {STATE_NAMES[INDIAN_STATES[i]]}",
        )
        state_code = INDIAN_STATES[state_idx]

    with col2:
        income = st.number_input(
            "Annual Income (₹)", min_value=0, max_value=100_000_000,
            value=200_000, step=10_000,
        )
        caste = st.selectbox("Category", ["general", "obc", "sc", "st", "minority"])
        occupation = st.text_input("Occupation", value="")
        disability = st.checkbox("Person with Disability")

    specific_query = st.text_input(
        "Any specific scheme or need? (optional)",
        placeholder="e.g., housing loan, education scholarship, crop insurance...",
    )

    if st.button("🔍 Find Schemes", type="primary"):
        profile = UserProfile(
            age=age,
            gender=gender,
            state_code=state_code,
            income=float(income),
            caste=caste,
            occupation=occupation if occupation else None,
            disability=disability,
        )
        set_user_profile(profile)

        query_text = specific_query if specific_query else (
            f"Find government schemes for a {age} year old {gender} "
            f"from {STATE_NAMES.get(state_code, state_code)} "
            f"with annual income ₹{income:,}, category {caste}"
        )

        query = UserQuery(
            text=query_text,
            user_lang="en",
            persona=Persona.CITIZEN,
            act_filter=ActFilter.ALL,
            user_profile=profile,
        )

        with st.spinner("Searching for eligible schemes..."):
            try:
                response: OrchestratorResponse = handle_fn(query)

                st.markdown("---")
                st.subheader("📋 Results")
                st.markdown(response.answer_text)

                if response.schemes:
                    st.subheader("Matching Schemes")
                    for i, s in enumerate(response.schemes, 1):
                        with st.expander(f"{i}. {s.get('name', 'Unnamed Scheme')}"):
                            st.markdown(f"**Description:** {s.get('description', 'N/A')}")
                            st.markdown(f"**Eligibility:** {s.get('eligibility', 'N/A')}")
                            st.markdown(f"**Benefits:** {s.get('benefits', 'N/A')}")
                            url = s.get("apply_url", "")
                            if url:
                                st.markdown(f"[Apply Here]({url})")

                if response.disclaimer:
                    st.caption(response.disclaimer)

            except Exception as exc:
                st.error(f"Error: {exc}")
