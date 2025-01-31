# main.py
import os
import yaml
from dataclasses import dataclass
from typing import List, Dict
import arxiv
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from typing import List
from langchain.agents import tool  
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor

@dataclass
class Config:
    data_dir: str
    model_dir: str
    vector_db_path: str
    arxiv_query: str = "cat:cs.CV AND submittedDate:[2023 TO 2024]"
    chunk_size: int = 512
    chunk_overlap: int = 100
    embedding_model: str = "all-MiniLM-L6-v2"

EXPLAIN_TEMPLATE = """Explain {concept} using:
1. Simple analogy: {analogy}
2. Mathematical formulation: {math}
3. Computer vision application: {application}

Context: {context}"""

class ARXIVRAGSystem:
    def __init__(self, config: Config):
        self.config = config
        os.makedirs(config.data_dir, exist_ok=True)
        os.makedirs(config.model_dir, exist_ok=True)
        self.llm = Ollama(base_url="http://localhost:8083", model="llama2-arxiv-4bit:latest", num_predict=500, temperature=0.5)
    
            

    def download_papers(self, max_results: int = 10) -> List[Dict]:
        """Fetch papers from arXiv CVPR category"""
        client = arxiv.Client()
        search = arxiv.Search(
            query=self.config.arxiv_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        papers = []
        for result in client.results(search):
            paper = {
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "pdf_url": result.pdf_url,
                "published": result.published.date().isoformat()
            }
            papers.append(paper)
            result.download_pdf(dirpath=self.config.data_dir)
        return papers

    def process_pdf(self, filename: str) -> str:
        """Extract and chunk text from PDF"""
        doc = fitz.open(os.path.join(self.config.data_dir, filename))
        text = " ".join([page.get_text() for page in doc])
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        return splitter.split_text(text)

    def create_vector_store(self):
        """Create FAISS vector database from papers"""
        embedder = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        texts = []
        for filename in os.listdir(self.config.data_dir):
            if filename.endswith(".pdf"):
                chunks = self.process_pdf(filename)
                texts.extend(chunks)
        
        # Use from_texts instead of from_embeddings
        db = FAISS.from_texts(texts, embedder)
        db.save_local(self.config.vector_db_path)
        return db

    def load_qa_chain(self):
        """Load retrieval QA chain with local LLM"""
        embedder = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        
        db = FAISS.load_local(
            self.config.vector_db_path,
            embedder,
            allow_dangerous_deserialization=True
        )
        
        llm = Ollama(base_url="http://localhost:8083", model="llama2-arxiv-4bit:latest", num_predict=500, temperature=0.5)

        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff"
        )
    
    
    def get_tools(self):
        """Create custom tools for agent"""
        base_retriever = self.load_qa_chain().retriever
        
        # @tool
        # def paper_summarizer(paper_title: str) -> str:
        #     """Summarize key contributions of a paper"""
        #     try:
        #         # Get documents with proper metadata filtering
        #         docs = base_retriever.invoke(
        #             paper_title,  # Direct text query
        #             filter={"title": paper_title},
        #             k=1  # Get top match only
        #         )
                
        #         if not docs:
        #             return "No paper found with that title"
                    
        #         # Access content safely
        #         content = docs[0].page_content[:2000]  # Truncate for LLM context
        #         return self.llm(f"Summarize in 3 bullet points: {content}")
                
        #     except Exception as e:
        #         return f"Error summarizing paper: {str(e)}"

        def safe_retrieve(query: str, filter: dict = None, k: int = 3):
            """Safe wrapper for retriever calls"""
            if isinstance(query, dict):  # Handle accidental dict inputs
                query = query.get("query", "")
            return base_retriever.invoke(query, filter=filter, k=k)

        @tool
        def paper_summarizer(paper_title: str) -> str:
            """Summarize key contributions of a paper"""
            docs = safe_retrieve(
                f"Summarize this paper: {paper_title}",
                filter={"title": paper_title},
                k=1
            )
            # Access content safely
            content = docs[0].page_content[:2000]  # Truncate for LLM context
            return self.llm(f"Summarize in 3 bullet points: {content}")
        
        
        @tool
        def paper_comparator(paper_titles: str) -> str:
            """Compare methodologies between papers. Input format: 'title1, title2'"""
            # Convert string to list
            paper_list = [t.strip() for t in paper_titles.split(",")]
            
            contexts = []
            for title in paper_list:
                docs = base_retriever.invoke({
                    "query": f"Extract methodology section for {title}",
                    "filter": {"title": title}
                })
                if docs:
                    contexts.append(f"Paper: {title}\nContent: {docs[0].page_content}")
            
            if not contexts:
                return "No relevant papers found"
            
            return self.llm(f"Compare methodologies:\n{'-'*50}\n" + "\n\n".join(contexts))
        
        @tool
        def methodology_explainer(concept: str, depth: str = "beginner") -> str:
            """Explain complex CV concepts at specified depth (beginner/intermediate/advanced)"""
            docs = base_retriever.invoke({
                "query": f"Find technical explanations of {concept}",
                "k": 3
            })
            contexts = [d.page_content for d in docs]
            
            # Use the structured template
            prompt = EXPLAIN_TEMPLATE.format(
                concept=concept,
                context="\n---\n".join(contexts)
            )
            
            return self.llm(
                f"Follow this format strictly:\n{prompt}\n"
                f"Adapt explanation for {depth} level:"
            )

        return [
            paper_summarizer,
            paper_comparator,
            methodology_explainer,
            Tool(
                name="General QA",
                func=self.load_qa_chain().run,
                description="Answer general questions about computer vision research"
            )
        ]

    def get_agent(self):
        prompt = hub.pull("hwchase17/react")
        prompt.template = """Answer questions using these tools: {tool_names}

    Strictly follow this format:
    Thought: Reason about the task
    Action: tool_name
    Action Input: "input"
    Observation: tool_result
    ... (repeat if needed)
    Thought: Final answer
    Final Answer: Concise response

    Begin!

    Question: {input}
    Thought:{agent_scratchpad}"""
        
        return AgentExecutor(
            agent=create_react_agent(self.llm, self.get_tools(), prompt),
            tools=self.get_tools(),
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=3,
            early_stopping_method="generate"
        )

    
    
def load_config(config_path: str) -> Config:
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    return Config(**config_data)

if __name__ == "__main__":
    
    config = load_config("config.yaml")
    system = ARXIVRAGSystem(config)
    
    # Initialize agent
    agent = system.get_agent()

    # query = "compare methodologies between 'A_Generalist_FaceX_via_Learning_Unified_Facial_Representation, Compressing_Deep_Image_Super_resolution_Models' for a beginner"
    query = "summarize A_Generalist_FaceX_via_Learning_Unified_Facial_Representation"
    print(f"\nQuery: {query}")
    print("Answer:", agent.invoke({"input": query}))

    
    # Example queries to try:
    # queries = [
    #     "Compare the methodologies in 'Attention Is All You Need' and 'ResNet' papers",
    #     "Explain transformer architectures to a beginner",
    #     "Summarize the key contributions of 'Mask R-CNN'"
    # ]
    
    # for query in queries:
    #     print(f"\nQuery: {query}")
    #     print("Answer:", agent.invoke({"input": query})["output"])