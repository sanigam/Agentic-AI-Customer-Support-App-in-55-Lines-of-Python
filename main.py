"""AI Learning Center Support - Streamlit Web Interface

A simple web UI for the multi-agent customer support system that:
- Accepts user questions about policies or general AI topics
- Suppresses verbose logging for cleaner output
- Extracts and displays the final answer from the agent response
"""

import logging
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import streamlit as st

# Configure logging to suppress verbose output from CrewAI
logging.basicConfig(level=logging.ERROR)
for logger_name in ("crewai", "crewai_tools"):
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from crew import get_support_response


def extract_final_answer(text: str) -> str:
    """Extract the final answer from CrewAI response if it contains a 'Final Answer:' marker.
    
    Args:
        text: Raw response text from the crew execution
        
    Returns:
        Cleaned final answer text
    """
    text = str(text).strip()
    
    # Look for "final answer:" marker (case insensitive) and extract text after it
    lower_text = text.lower()
    marker = "final answer:"
    
    if marker in lower_text:
        # Find the last occurrence and extract everything after it
        index = lower_text.rfind(marker)
        text = text[index + len(marker):].strip()
    
    # Remove common leading separators
    return text.lstrip("#- ")


# Configure Streamlit page
st.set_page_config(page_title="AI Learning Center Support", page_icon="🤖")

# Page header
st.title("AI Learning Center Support")
st.write("Ask questions about our policies, AI concepts, or current tech trends.")

# User input
user_query = st.text_area(
    "Your question",
    placeholder="e.g., What is your refund policy? or What is machine learning?",
    height=100
)

# Process query when button is clicked
if st.button("Get Answer", type="primary"):
    query = user_query.strip()
    
    if not query:
        st.warning("⚠️ Please enter a question.")
    else:
        with st.spinner("🤔 Our AI team is working on your question..."):
            # Suppress verbose output from CrewAI agents
            with StringIO() as stdout_buffer, StringIO() as stderr_buffer:
                with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                    raw_response = get_support_response(query)
            
            # Extract clean answer from the response
            final_answer = extract_final_answer(raw_response)
        
        # Display the result
        st.success("✅ Answer ready!")
        st.subheader("Answer")
        st.write(final_answer)
