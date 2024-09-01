import streamlit as st
from streamlit_supabase_auth import login_form, logout_button

session = login_form(
    url="https://xxxx.supabase.co",
    apiKey="<SUPABASE_KEY>",
    providers=["apple", "facebook", "github", "google"],
)


# Update query param to reset url fragments
st.experimental_set_query_params(page=["success"])
with st.sidebar:
    st.write(f"Welcome {session['user']['email']}")
    logout_button()
