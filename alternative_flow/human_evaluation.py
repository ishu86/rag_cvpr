import streamlit as st
import pandas as pd

def evaluation_dashboard():
    st.title("Response Quality Evaluation")
    
    # Load evaluation data
    eval_data = pd.read_csv("evaluation_results.csv")
    
    # Sidebar filters
    selected_metric = st.sidebar.selectbox("Select Metric", 
                                         ["Relevance", "Coherence", "Accuracy"])
    
    # Metric visualization
    st.subheader(f"{selected_metric} Scores")
    avg_score = eval_data[selected_metric.lower()].mean()
    st.metric("Average Score", f"{avg_score:.2%}")
    
    # Manual grading interface
    st.subheader("Grade Responses")
    sample = eval_data.sample(1).iloc[0]
    st.write(f"**Query:** {sample['query']}")
    st.write(f"**Response:** {sample['response']}")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        relevance = st.slider("Relevance (1-5)", 1, 5, 3)
    with col2:
        coherence = st.slider("Coherence (1-5)", 1, 5, 3)
    with col3: 
        accuracy = st.slider("Accuracy (1-5)", 1, 5, 3)
    
    if st.button("Submit Evaluation"):
        # Update dataset
        new_entry = pd.DataFrame([{
            "query": sample["query"],
            "response": sample["response"],
            "relevance": relevance,
            "coherence": coherence,
            "accuracy": accuracy
        }])
        new_entry.to_csv("evaluation_results.csv", mode="a", header=False)
        st.success("Evaluation submitted!")