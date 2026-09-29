"""Starter Streamlit dashboard for the EM630 project."""

import streamlit as st

# WHAT: Display the project title and setup status.
# WHY: Provide an entry point before modelling starts.
# HOW: Use simple text elements; later pages will read saved outputs.
# OUTPUT: A starter browser page without model training.
st.title("EM630 Power Utility Consumer Analytics")
st.info("Project structure is ready. Analysis and prediction pages will follow.")
st.caption("Public project data are synthetic.")
