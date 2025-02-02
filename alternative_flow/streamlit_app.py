import streamlit as st
from main import ARXIVRAGSystem, load_config
from dataclasses import asdict
import yaml

# Load configuration
config = load_config("config.yaml")

# Set page config
st.set_page_config(
    page_title="arXiv CVPR Research Assistant",
    page_icon="📚",
    layout="centered"
)

# Initialize system
@st.cache_resource
def init_system():
    return ARXIVRAGSystem(config)

system = init_system()

# Sidebar for setup
with st.sidebar:
    st.header("⚙️ System Setup")
    if st.button("Initialize System (Download Papers)"):
        with st.spinner("Downloading papers and creating vector store..."):
            system.download_papers(max_results=10)
            system.create_vector_store()
        st.success("System initialized successfully!")
    
    st.markdown("---")
    st.markdown("**Example Questions:**")
    example_questions = [
        "Summarize A Generalist FaceX via Learning Unified Facial Representation",
        "Compare transformer architectures and CNN approaches",
        "Explain contrastive learning to a beginner",
        "What's the latest in 3D object detection?",
        "Compare diffusion models and GANs for image generation"
    ]
    
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state.current_question = q

# Main interface
st.title("📚 arXiv CVPR Research Assistant")
st.markdown("Explore CVPR papers from arXiv with AI-powered analysis!")

# Question input
question = st.text_area(
    "Ask your research question:",
    value=st.session_state.get("current_question", ""),
    height=100
)

# Response container
response_container = st.container()

if st.button("Analyze Papers", type="primary") or "current_question" in st.session_state:
    if not question:
        st.warning("Please enter a question")
        st.stop()
    
    with st.spinner("Analyzing papers..."):
        try:
            agent = system.get_agent()
            context = system.get_context(question)
            result = agent.invoke({"input": question, "context": context})
            
            with response_container:
                st.markdown("### Analysis Results")
                
                # Handle both parsed and unparsed outputs
                if "output" in result:
                    st.markdown("#### Answer")
                    st.markdown(result["output"])
                elif isinstance(result, str):  # Fallback for unparsed output
                    st.markdown("#### Answer")
                    st.markdown(result)
                
                if "intermediate_steps" in result:
                    st.markdown("#### Research Process")
                    for step in result["intermediate_steps"]:
                        if isinstance(step, tuple) and len(step) == 2:
                            action, observation = step
                            with st.expander(f"{action.tool}: {action.tool_input}"):
                                st.markdown(f"**Observation**:\n{observation}")
                
                st.markdown("---")
                st.markdown("**Retrieved Context**")
                st.info(context[:2000])  # Show first 2000 chars of context
                
        except Exception as e:
            st.error(f"Error processing request: {str(e)}")
            st.error("Please ensure you've initialized the system in the sidebar")

# Footer
st.markdown("---")
st.markdown("""
**Tips:**
- Use exact paper titles from context for best results
- Start with general questions, then drill down into specifics
- Click example questions in sidebar to try different analysis types
""")
