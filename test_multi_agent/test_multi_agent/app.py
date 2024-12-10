# app.py

import streamlit as st
import asyncio
from core.agent_manager import AgentManager

# Set page configuration with custom theme
st.set_page_config(
    page_title="Medical AI Agents",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        padding: 1.5rem;
        margin-bottom: 1rem;
        font-weight: bold;
        background: linear-gradient(120deg, #1E88E5 0%, #1565C0 100%);
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(30,136,229,0.2);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #0D47A1;
        padding: 0.5rem 0;
        border-bottom: 2px solid #1E88E5;
        margin-bottom: 1rem;
    }
    .task-container {
        background-color: #F8F9FA;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .validation-box {
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        border: none;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #1565C0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .stTextArea>div>div {
        border-radius: 8px;
        border: 2px solid #E3F2FD;
    }
    .sidebar-content {
        padding: 1rem;
        background-color: #F8F9FA;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def get_agent_manager():
    return AgentManager()

def show_results_page(result_data):
    st.markdown("<h1 class='main-header'>Final Validated Content</h1>", unsafe_allow_html=True)
    
    # Add a subheader based on the content type
    if "summary" in result_data["result"]:
        st.markdown("<h2 class='sub-header'>Medical Text Summary</h2>", unsafe_allow_html=True)
    elif "article" in result_data["result"]:
        st.markdown("<h2 class='sub-header'>Research Article</h2>", unsafe_allow_html=True)
    elif "sanitized_data" in result_data["result"]:
        st.markdown("<h2 class='sub-header'>Redacted PHI Content</h2>", unsafe_allow_html=True)
    
    # Display content in a styled box
    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
    if "summary" in result_data["result"]:
        st.write(result_data["result"]["summary"])
    elif "article" in result_data["result"]:
        st.write(result_data["result"]["article"])
    elif "sanitized_data" in result_data["result"]:
        st.write(result_data["result"]["sanitized_data"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Action buttons in columns
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # Export button
        if st.button("📥 Export Results"):
            export_data = ""
            if "summary" in result_data["result"]:
                export_data = result_data["result"]["summary"]
            elif "article" in result_data["result"]:
                export_data = result_data["result"]["article"]
            elif "sanitized_data" in result_data["result"]:
                export_data = result_data["result"]["sanitized_data"]
                
            st.download_button(
                label="💾 Download Content",
                data=export_data,
                file_name="final_content.txt",
                mime="text/plain"
            )
    
    with col3:
        # Return button
        if st.button("🏠 Return to Main Page"):
            st.session_state.show_results = False
            st.rerun()

def main():
    # Sidebar styling
    with st.sidebar:
        st.markdown("<h2 style='text-align: center; color: #1E88E5;'>Tasks</h2>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-content'>", unsafe_allow_html=True)
        task_type = st.radio(
            "",  # Empty label as we're using custom header
            ["summarize", "write_article", "Redact PHI"],
            format_func=lambda x: {
                "summarize": "📝 Summarize Medical Text",
                "write_article": "📚 Write Research Article",
                "Redact PHI": "🔒 Redact PHI"
            }[x]
        )
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Main content - Single header for the entire page
    st.markdown("<h1 class='main-header'>Medical Multi-Agent System</h1>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'result_data' not in st.session_state:
        st.session_state.result_data = None
    
    if st.session_state.show_results:
        show_results_page(st.session_state.result_data)
        return
    
    agent_manager = get_agent_manager()
    
    # Task containers with consistent styling
    st.markdown("<div class='task-container'>", unsafe_allow_html=True)
    
    if task_type == "summarize":
        st.markdown("<h2 class='sub-header'>📝 Summarize Medical Text</h2>", unsafe_allow_html=True)
        input_text = st.text_area("Enter medical text to summarize", height=200)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Generate Summary"):
                with st.spinner("Processing..."):
                    result = asyncio.run(agent_manager.process_task("summarize", input_text))
                    st.session_state.result_data = result
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.subheader("Summary")
                    st.write(result["result"]["summary"])
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<div class='validation-box'>", unsafe_allow_html=True)
                    st.subheader("Validation")
                    
                    # Extract and display score
                    feedback = result["validation"]["feedback"]
                    if "Score:" in feedback:
                        score = feedback.split("Score:")[1].split("\n")[0].strip()
                        st.markdown(f"""
                            <div style='background-color: #E3F2FD; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
                                <h3 style='margin: 0; color: #1565C0; text-align: center;'>Validation Score: {score}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.write(feedback)
                    st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if st.session_state.result_data and st.button("👁️ View Edited Content"):
                st.session_state.show_results = True
                st.rerun()
    
    elif task_type == "write_article":
        st.markdown("<h2 class='sub-header'>📚 Write Research Article</h2>", unsafe_allow_html=True)
        topic = st.text_input("Enter research topic")
        key_points = st.text_area("Enter key points (one per line)", height=150)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📝 Generate Article"):
                with st.spinner("Processing..."):
                    input_data = {"topic": topic, "key_points": key_points}
                    result = asyncio.run(agent_manager.process_task("write_article", input_data))
                    st.session_state.result_data = result
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.subheader("Article")
                    st.write(result["result"]["article"])
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<div class='validation-box'>", unsafe_allow_html=True)
                    st.subheader("Validation")
                    
                    # Extract and display score
                    feedback = result["validation"]["feedback"]
                    if "Score:" in feedback:
                        score = feedback.split("Score:")[1].split("\n")[0].strip()
                        st.markdown(f"""
                            <div style='background-color: #E3F2FD; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
                                <h3 style='margin: 0; color: #1565C0; text-align: center;'>Validation Score: {score}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.write(feedback)
                    st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if st.session_state.result_data and st.button("👁️ View Edited Content"):
                st.session_state.show_results = True
                st.rerun()
    
    elif task_type == "Redact PHI":
        st.markdown("<h2 class='sub-header'>🔒 Redact Protected Health Information (PHI)</h2>", unsafe_allow_html=True)
        input_text = st.text_area("Enter medical text to redact PHI", height=200)
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔐 Redact PHI"):
                with st.spinner("Processing..."):
                    result = asyncio.run(agent_manager.process_task("sanitize", input_text))
                    st.session_state.result_data = result
                    st.markdown("<div class='result-box'>", unsafe_allow_html=True)
                    st.subheader("Redacted Text")
                    st.write(result["result"]["sanitized_data"])
                    st.markdown("</div>", unsafe_allow_html=True)
                    st.markdown("<div class='validation-box'>", unsafe_allow_html=True)
                    st.subheader("Validation")
                    
                    # Extract and display score
                    feedback = result["validation"]["feedback"]
                    if "Score:" in feedback:
                        score = feedback.split("Score:")[1].split("\n")[0].strip()
                        st.markdown(f"""
                            <div style='background-color: #E3F2FD; padding: 1rem; border-radius: 8px; margin-bottom: 1rem;'>
                                <h3 style='margin: 0; color: #1565C0; text-align: center;'>Validation Score: {score}</h3>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.write(feedback)
                    st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            if st.session_state.result_data and st.button("👁️ View Edited Content"):
                st.session_state.show_results = True
                st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main() 