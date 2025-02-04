import os
import re
import yaml
import ollama
import numpy as np
from dataclasses import dataclass
from typing import List, Dict
import arxiv
from langchain.docstore.document import Document
import argparse
import fitz
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import OllamaEmbeddings
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain import hub
from typing import List
from langchain.agents import tool  
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from monitoring import monitor_resources

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
        self.llm = OllamaLLM(base_url="http://localhost:8083", model="llama2:latest", num_predict=500, temperature=0.5)
        self.embed_model = OllamaLLM(base_url="http://localhost:8083", model="nomic-embed-text:latest")
    
            

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
    
    def is_arxiv_id(self, input_str: str) -> bool:
        """Check if input matches arXiv ID pattern"""
        return re.match(r'^\d{4}\.\d{5}(v\d+)?$', input_str) is not None

    def normalize_input(self, paper_input: str) -> dict:
        """Return filter criteria based on input type"""
        if self.is_arxiv_id(paper_input):
            return {"field": "arxiv_id", "value": paper_input}
        else:
            return {
                "field": "title_lower",
                "value": paper_input.replace("_", " ").strip().lower()
            }
    @monitor_resources
    def process_pdf(self, filename: str) -> str:
        """Extract and chunk text from PDF"""
        doc = fitz.open(os.path.join(self.config.data_dir, filename))
        text = " ".join([page.get_text() for page in doc])
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        return splitter.split_text(text) 
    
    def clean_paper_title(self, filename: str) -> str:
        """Clean arxiv paper filename to get readable title"""
        # Remove arxiv ID and version number
        import re
        parts = re.split(r'\d+v\d+\.', filename)
        clean_title = parts[-1]
        
        # Remove .pdf extension
        clean_title = clean_title.rsplit('.', 1)[0]
        
        # Replace underscores with spaces 
        clean_title = clean_title.replace('_', ' ')
        
        # Clean extra whitespace
        clean_title = ' '.join(clean_title.split())
        
        return clean_title
        
    def extract_arxiv_id(self, filename: str) -> str:
        """Extract arXiv ID from filename using regex"""
        pattern = r"(\d{4}\.\d{5})|arXiv:(\d{4}\.\d{5})"
        match = re.search(pattern, filename)
        return match.group(1) if match and match.group(1) else match.group(2) if match else None


    @monitor_resources
    def create_vector_store(self):
        embedder = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:8083")
        
        documents = []
        for filename in os.listdir(self.config.data_dir):
            if filename.endswith(".pdf"):
                clean_title = self.clean_paper_title(filename)
                arxiv_id = self.extract_arxiv_id(filename)
                
                chunks = self.process_pdf(filename)
                documents.extend([
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": filename,
                            "title": clean_title,
                            "title_lower": clean_title.lower(),  # For case-insensitive search
                            "arxiv_id": arxiv_id   # Fallback if no match
                        }
                    ) for chunk in chunks
                ])
        
        db = FAISS.from_documents(documents, embedder)
        db.save_local(self.config.vector_db_path)
        return db
  

    @monitor_resources
    def load_qa_chain(self):
        """Load retrieval QA chain with local LLM"""
        embedder = OllamaEmbeddings(
        model="nomic-embed-text:latest",  # Model name in Ollama
        base_url="http://localhost:8083"  # Ollama server URL
    )
        
        db = FAISS.load_local(
            self.config.vector_db_path,
            embedder,
            allow_dangerous_deserialization=True
        )
      

        
        llm = Ollama(base_url="http://localhost:8083", model="llama2-arxiv-4bit:latest", num_predict=500, temperature=0.5)


        
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=db.as_retriever(search_kwargs={"k": 5}),
            chain_type="stuff"
        )
    
    
    def get_context(self, query: str) -> str:
        """Retrieve relevant context for a query"""
        retriever = self.load_qa_chain().retriever
        docs = retriever.invoke(query)
        return "\n".join([d.page_content for d in docs][:5])
    
    def get_tools(self):
        """Create custom tools for agent"""
        base_retriever = self.load_qa_chain().retriever
        
        @tool
        def paper_summarizer(paper_input: str) -> str:
            """Summarize a paper using title or arXiv ID. Input format: 'Title' or 'arXiv ID'"""
            try:
                criteria = self.normalize_input(paper_input)
                docs = base_retriever.invoke(
                    f"Summary of {paper_input}",
                    filter={criteria["field"]: criteria["value"]},
                    k=2
                )
                
                if not docs:
                    return f"Paper '{paper_input}' not found. Try arXiv ID (e.g., 2401.00608v5) or exact title"
                    
                # Get display title from metadata
                display_title = docs[0].metadata.get("title", paper_input)
                content = "\n".join([d.page_content[:1000] for d in docs])
                return self.llm(f"Summarize in 3 bullet points: {content}\nPaper Title: {display_title}")
            
            except Exception as e:
                return f"Summarization error: {str(e)}"
                
        
        @tool
        def paper_comparator(paper_inputs: str) -> str:
            """Compare 2 papers. Input format: 'Title/ID1, Title/ID2'"""
            try:
                papers = [p.strip() for p in paper_inputs.split(",", 1)]
                if len(papers) != 2:
                    return "Error: Format must be 'Title/ID1, Title/ID2'"
                
                contexts = []
                for paper in papers:
                    criteria = self.normalize_input(paper)
                    docs = base_retriever.invoke(
                        f"Methodology of {paper}",
                        filter={criteria["field"]: criteria["value"]},
                        k=2
                    )
                    if docs:
                        title = docs[0].metadata.get("title", paper)
                        contexts.append(f"=== {title} ===\n{docs[0].page_content}")
                
                if len(contexts) != 2:
                    return "Could not find both papers"
                    
                return self.llm(f"Compare these methodologies:\n{'-'*40}\n" + "\n\n".join(contexts))
            
            except Exception as e:
                return f"Comparison error: {str(e)}"
        
        @tool
        def methodology_explainer(concept: str, depth: str = "beginner") -> str:
            """Explain concepts with paper citations. Input format: 'concept, depth'"""
            try:
                docs = base_retriever.invoke(
                    f"Technical explanations of {concept}",
                    k=4
                )
                
                # Build citation map
                sources = {}
                for d in docs:
                    title = d.metadata.get("title", "Unknown Paper")
                    sources[title] = d.metadata.get("arxiv_id", "")
                
                # Generate explanation
                explanation = self.llm(
                    f"Explain {concept} at {depth} level using:\n"
                    f"{' '.join([d.page_content for d in docs])}\n\n"
                    "Include citations in format [PaperTitle]"
                )
                
                # Add reference section
                refs = "\n".join([f"- [{title}](https://arxiv.org/abs/{arxiv_id})" 
                                for title, arxiv_id in sources.items()])
                return f"{explanation}\n\nReferences:\n{refs}"
            
            except Exception as e:
                return f"Explanation error: {str(e)}"

        tools = [
            Tool(
                name="paper_summarizer",
                func=paper_summarizer,
                description="Summarize a paper using title or arXiv ID"
            ), Tool(
                name="paper_comparator",
                func=paper_comparator,
                description="Compare 2 papers. Input format: 'Title/ID1, Title/ID2'"
            ), Tool(
                name="methodology_explainer",
                func=methodology_explainer,
                description="Explain concepts with paper citations. Input format: 'concept, depth'"
            )
        ]
        
        return tools    


 
    @monitor_resources
    def get_agent(self):
        PROMPT_TEMPLATE = """Follow STRICTLY:
    1. If asked about a SINGLE paper → paper_summarizer
    2. If comparing papers → paper_comparator
    3. For concept explanations → methodology_explainer
    
    FORMAT:
    Thought: [Identify tool based on question type]
    Action: {tool_names}
    Action Input: "[EXACT title/ID from context]"
    Observation: [Tool output]
    Final Answer: [Concise response with citations]
    
    Context: {context}
    Question: {input}
    {agent_scratchpad}"""
    
        tools = self.get_tools()
        tool_names = ", ".join([t.name for t in tools])
        tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).partial(tools=tools,
            tool_names=tool_names,
            tool_descriptions=tool_descriptions
        )
    
        return AgentExecutor(
            agent=create_react_agent(self.llm, tools, prompt),
            tools=tools,
            verbose=True,
            handle_parsing_errors=lambda _: "Format error - try rephrasing",
            max_iterations=2,
            return_intermediate_steps=True,
            output_parser=UniversalOutputParser(),
            validate_tools=True 
        )
        
class UniversalOutputParser(ReActSingleInputOutputParser):
    def parse(self, text: str):
        try:
            # First try strict format parsing
            return super().parse(text)
        except Exception:
            # Fallback 1: Extract content between Answer: and Human:
            answer_section = re.search(r"Answer:(.*?)(Human:|$)", text, re.DOTALL)
            if answer_section:
                return {"output": answer_section.group(1).strip()}
            
            # Fallback 2: Capture everything after first paragraph
            full_answer = re.sub(r"^.*?(?=\bAnswer:)", "", text, flags=re.DOTALL)
            return {"output": full_answer.strip() or text[:1000]}
    
    
def load_config(config_path: str) -> Config:
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    return Config(**config_data)

import argparse  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Initialize the system and create vector store")
    args = parser.parse_args()

    config = load_config("config.yaml")
    system = ARXIVRAGSystem(config)

    if args.setup:
        print("Downloading papers...")
        system.download_papers(max_results=10)  # Adjust number of papers as needed
        
        print("Creating vector store...")
        system.create_vector_store()
        print("Setup complete! Vector store created at:", config.vector_db_path)
    else:
        # Normal operation
        agent = system.get_agent()
        query = "What are large language models? Answer up to the point"
        context = system.get_context(query)
        print(f"\nContext:\n{context}")
        print(f"\nQuery: {query}")
        print("Answer:", agent.invoke({"input": query, "context": context})['output'])


if __name__ == "__main__":
    main()
