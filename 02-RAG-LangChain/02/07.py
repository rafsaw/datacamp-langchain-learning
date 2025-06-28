from ragas.metrics import context_precision, faithfulness
from ragas import evaluate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from datasets import Dataset
from langchain_chroma import Chroma
from langsmith.evaluation import LangChainStringEvaluator 
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Set up API keys and LLM
openai_api_key = os.getenv("OPENAI_API_KEY")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

predicted_answer = "RAG improves question answering with LLMs by generating correct answers even when the correct answer is not present in any retrieved document, achieving a notable accuracy of 11.8% in such cases, while extractive models would score 0%. Additionally, RAG models outperform other models like BART in terms of generating factually correct and diverse text, as well as being able to answer questions in a more flexible, abstractive manner rather than relying solely on extractive methods."
ref_answer = "Retrieval-Augmented Generation (RAG) improves question answering with large language models (LLMs) by combining a retrieval mechanism with a generative model. The retrieval system fetches relevant documents or passages from external knowledge sources, giving the LLM access to more up-to-date and accurate information than what it has learned during training. This allows RAG to generate responses that are grounded in factual data, reducing the risk of hallucination and improving the model's accuracy, especially in niche or specialized domains where the LLM alone may lack expertise. By leveraging both external knowledge and the generative abilities of LLMs, RAG enhances the quality, relevance, and factuality of the answers provided."
query = "How does RAG improve question answering with LLMs?"


prompt = """You are an expert professor specialized in grading students' answers to questions.
You are grading the following question:{query}
Here is the real answer:{answer}
You are grading the following predicted answer:{result}
Respond with CORRECT or INCORRECT:
Grade:"""

# Convert the string into a chat prompt template
# prompt_template =  ChatPromptTemplate.from_template(prompt)

prompt_template = PromptTemplate( 
    input_variables=["query", "answer", "result"], 
    template=prompt 
) 
eval_llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", api_key=openai_api_key)

qa_evaluator = LangChainStringEvaluator( 
    "qa", 
    config={ 
    "llm": eval_llm,
    "prompt": prompt_template 
    } 
) 

score = qa_evaluator.evaluator.evaluate_strings( 
    prediction=predicted_answer, 
    reference=ref_answer, 
    input=query 
) 

print(f"Score: {score}") 