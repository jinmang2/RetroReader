import streamlit as st

import io
import os
import yaml

# from ..retro_reader import RetroReader

from confirm_button_hack import cache_on_button_press
from constants import (
    ROOT_PASSWORD,
    QUERY_HELP_TEXT,
    CONTEXT_HELP_TEXT,
    EXAMPLE_QUERY,
    EXAMPLE_CONTEXTS
)

# SETTING PAGE CONFIG TO WIDE MODE
st.set_page_config(layout="wide")


def main():
    st.title("Retrospective Reader Demo")
    
    with st.form(key="my_form"):
        query = st.text_input(
            label="Type your query",
            value=EXAMPLE_QUERY,
            max_chars=None,
            help=QUERY_HELP_TEXT,
        )
        context = st.text_area(
            label="Type your context",
            value=EXAMPLE_CONTEXTS,
            height=300,
            max_chars=None,
            help=CONTEXT_HELP_TEXT,
        )
        submit_button = st.form_submit_button(label="Submit")
        
    if submit_button:
        st.write(query, context)
    

@cache_on_button_press('Authenticate')
def authenticate(password: str) -> bool:
    print(type(password))
    return password == ROOT_PASSWORD


password = st.text_input('password', type="password")

if authenticate(password):
    st.success("You are authenticated!")
    main()
else:
    st.error("The password is invalid.")