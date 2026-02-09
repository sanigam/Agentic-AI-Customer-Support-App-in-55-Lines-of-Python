"""AI Learning Center Support - Streamlit Web Interface

A simple web UI for the multi-agent customer support system that:
- Accepts user questions about policies or general AI topics
- Suppresses verbose logging for cleaner output
- Extracts and displays the final answer from the agent response
"""
import streamlit as st
from crew import support_crew # Import the crew we just built

st.title("AI Learning Center Support")
query = st.text_input("Ask a question:")

if st.button("Get Answer") and query:
    with st.spinner("Processing..."):
        result = support_crew.kickoff(inputs={'user_query': query})
        st.markdown(result)