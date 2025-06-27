import os
from datetime import datetime
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough

class ConfigurableRAG:
    def __init__(self, persist_directory, openai_api_key):
        self.openai_api_key = openai_api_key
        
        # Load database and create retriever
        embedding_function = OpenAIEmbeddings(api_key=openai_api_key, model='text-embedding-3-small')
        self.db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
        self.retriever = self.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        
        # Create LLM
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        
        # Define prompt templates
        self.prompts = {
            "restrictive": """
            Answer the following question using ONLY the context provided below. 
            Do not use any external knowledge or information not explicitly stated in the context.
            If the context doesn't contain enough information to answer the question, say so.

            Context:
            {context}

            Question:
            {question}

            Answer (based only on provided context):
            """,
            
            "permissive": """
            Answer the following question using both the provided context AND your general knowledge.
            Use the context as the primary source, but feel free to supplement with relevant 
            additional information, examples, or broader insights about the topic.

            Context from documents:
            {context}

            Question:
            {question}

            Comprehensive answer (context + general knowledge):
            """,
            
            "balanced": """
            Answer the following question primarily using the provided context, but you may 
            add relevant general knowledge if it directly supports or clarifies the context.
            Clearly distinguish between information from the context vs. your general knowledge.

            Context:
            {context}

            Question:
            {question}

            Answer:
            """
        }
    
    def format_docs(self, docs):
        """Convert list of documents to single string"""
        return "\n\n".join(doc.page_content for doc in docs)
    
    def create_chain(self, mode="restrictive"):
        """Create RAG chain with specified mode"""
        if mode not in self.prompts:
            raise ValueError(f"Mode must be one of: {list(self.prompts.keys())}")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("human", self.prompts[mode])
        ])
        
        return (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | self.llm
        )
    
    def ask(self, question, mode="restrictive", verbose=False, save_to_file=False, output_file=None):
        """Ask a question with specified mode and optionally save to file"""
        if verbose:
            print(f"Mode: {mode.upper()}")
            print(f"Question: {question}")
            print("-" * 50)
        
        chain = self.create_chain(mode)
        response = chain.invoke(question)
        
        if verbose:
            print(f"Answer: {response.content}")
            print("=" * 50)
        
        # Save to file if requested
        if save_to_file:
            self.save_response_to_file(question, response.content, mode, output_file)
        
        return response.content
    
    def save_response_to_file(self, question, answer, mode, output_file=None):
        """Save question and answer to a text file"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"rag_responses_{timestamp}.txt"
        
        # Create response entry
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = "=" * 80
        
        response_entry = f"""
{separator}
TIMESTAMP: {timestamp}
MODE: {mode.upper()}
QUESTION: {question}

ANSWER:
{answer}
{separator}

"""
        
        # Append to file
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(response_entry)
        
        print(f"Response saved to: {output_file}")
        return output_file
    
    def ask_and_compare_all_modes(self, question, save_to_file=True, output_file=None):
        """Ask the same question in all modes and optionally save comparison"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"rag_comparison_{timestamp}.txt"
        
        results = {}
        modes = ["restrictive", "balanced", "permissive"]
        
        print(f"Comparing all modes for question: {question}")
        print("=" * 60)
        
        for mode in modes:
            print(f"\n--- {mode.upper()} MODE ---")
            answer = self.ask(question, mode=mode, verbose=True)
            results[mode] = answer
        
        if save_to_file:
            self.save_comparison_to_file(question, results, output_file)
        
        return results
    
    def save_comparison_to_file(self, question, results, output_file):
        """Save comparison of all modes to file"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        separator = "=" * 80
        mode_separator = "-" * 40
        
        content = f"""
{separator}
RAG SYSTEM MODE COMPARISON
{separator}
TIMESTAMP: {timestamp}
QUESTION: {question}

"""
        
        for mode, answer in results.items():
            content += f"""
{mode_separator}
MODE: {mode.upper()}
{mode_separator}
ANSWER:
{answer}

ANSWER LENGTH: {len(answer)} characters
WORD COUNT: {len(answer.split())} words

"""
        
        content += f"""
{separator}
SUMMARY STATISTICS:
{separator}
"""
        
        for mode, answer in results.items():
            content += f"{mode.upper()} - Length: {len(answer)} chars, Words: {len(answer.split())}\n"
        
        content += f"\n{separator}\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\nComparison saved to: {output_file}")
        return output_file

# Usage example
if __name__ == "__main__":
    # Initialize the configurable RAG system
    openai_api_key = os.getenv("OPENAI_API_KEY")
    rag_system = ConfigurableRAG("rag_vs_fine_tuning_db", openai_api_key)
    
    question = "What are the main advantages of RAG over fine-tuning?"
    
    # Example 1: Single question with file saving
    print("=== SINGLE QUESTION WITH FILE SAVING ===")
    answer = rag_system.ask(
        question, 
        mode="permissive", 
        verbose=True, 
        save_to_file=True,
        output_file="single_response.txt"
    )
    
    # Example 2: Compare all modes and save comparison
    print("\n=== COMPARING ALL MODES ===")
    results = rag_system.ask_and_compare_all_modes(
        question, 
        save_to_file=True, 
        output_file="mode_comparison.txt"
    )
    
    # Example 3: Multiple questions saved to same file
    print("\n=== MULTIPLE QUESTIONS TO SAME FILE ===")
    questions = [
        "What are the computational requirements for fine-tuning?",
        "How does RAG handle real-time information?",
        "What are the cost differences between RAG and fine-tuning?"
    ]
    
    session_file = f"rag_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    for q in questions:
        rag_system.ask(
            q, 
            mode="balanced", 
            verbose=True, 
            save_to_file=True,
            output_file=session_file
        )
    
    print(f"\nAll responses saved to: {session_file}")
    
    # Example 4: Batch processing with different modes
    print("\n=== BATCH PROCESSING ===")
    batch_questions = [
        ("What is RAG?", "permissive"),
        ("Compare RAG and fine-tuning costs", "permissive"),
        ("What are the limitations of each approach?", "permissive")
    ]
    
    batch_file = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    for question, mode in batch_questions:
        rag_system.ask(
            question, 
            mode=mode, 
            verbose=True, 
            save_to_file=True,
            output_file=batch_file
        )