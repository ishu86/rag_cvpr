import pytest
from main import ARXIVRAGSystem, load_config
from evaluation_metrics import Evaluator

@pytest.fixture
def test_system():
    config = load_config("config.yaml")
    return ARXIVRAGSystem(config)

test_cases = [
    {
        "query": "Summarize A Generalist FaceX via Learning Unified Facial Representation",
        "expected_keywords": ["facial editing", "unified representation", "FRC"],
        "min_similarity": 0.65
    },
    {
        "query": "Compare CNN and Transformer architectures",
        "expected_keywords": ["attention", "convolution", "computational complexity"],
        "min_similarity": 0.6
    }
]

def test_rag_system(test_system):
    evaluator = Evaluator(test_system.embed_model)
    
    for case in test_cases:
        agent = test_system.get_agent()
        context = test_system.get_context(case["query"])
        result = agent.invoke({"input": case["query"], "context": context})
        
        # Relevance Check
        assert any(kw in result["output"].lower() for kw in case["expected_keywords"])
        
        # Coherence Check
        assert evaluator.semantic_similarity(result["output"], context) > case["min_similarity"]
        
        # Citation Accuracy
        assert evaluator.citation_relevance(result["output"], context) > 0.5