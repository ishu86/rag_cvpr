# Run automated tests
pytest test_suite.py -v > test_results.log

# Generate metric reports
python -c "
from evaluation_metrics import Evaluator
from main import ARXIVRAGSystem, load_config
import pandas as pd

system = ARXIVRAGSystem(load_config('config.yaml'))
evaluator = Evaluator(system.embed_model)

df = pd.read_csv('test_queries.csv')
results = []

for _, row in df.iterrows():
    agent = system.get_agent()
    context = system.get_context(row['query'])
    result = agent.invoke({'input': row['query'], 'context': context})
    
    results.append({
        'query': row['query'],
        'response': result['output'],
        'relevance': evaluator.semantic_similarity(result['output'], context),
        'citation_accuracy': evaluator.citation_relevance(result['output'], context),
        'hallucination_score': evaluator.hallucination_score(result['output'], context)
    })

pd.DataFrame(results).to_csv('evaluation_report.csv', index=False)"

# Open human evaluation interface
streamlit run human_evaluation.py