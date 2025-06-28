from ragas.metrics import context_precision, faithfulness
from ragas import evaluate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from datasets import Dataset
from langchain_chroma import Chroma

# Set up API keys and LLM
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

# Load existing vector store
vector_store = Chroma(
    persist_directory="02-RAG-LangChain/rag_store_02",
    embedding_function=embeddings
)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

retrieved_docs = [doc.page_content for doc in retriever.invoke("What are knowledge-intensive tasks?")]
print(retrieved_docs)

# contexts are the chunks returned by search (retriever).

# Create evaluation dataset
eval_dataset = Dataset.from_dict({
    "question": ["What are knowledge-intensive tasks?"],
    "contexts": [retrieved_docs],
    "ground_truth": ["Knowledge-intensive tasks are those that rely on deep expertise or domain-specific knowledge."]
})

# Run evaluation
# results = evaluate(
#     eval_dataset,
#     metrics=[context_precision],
#     llm=llm,
#     embeddings=embeddings
# )

# print("Evaluation Context Precision Results:")
# print(results)

prompt = f"""Answer the following question using only the provided context:

Context:
{chr(10).join(retrieved_docs)}

Question:
What are knowledge-intensive tasks?
"""

response = llm.invoke(prompt)

eval_dataset = Dataset.from_dict({
    "question": ["What are knowledge-intensive tasks?"],
    "contexts": [retrieved_docs],
    "ground_truth": ["Knowledge-intensive tasks are those that rely on deep expertise or domain-specific knowledge."],
    "response": [response.content]
})

results = evaluate(
    eval_dataset,
    metrics=[faithfulness],
    llm=llm,
    embeddings=embeddings
)

print("Evaluation Faithfulness Results:")
print(results)


# | RAGAS Metric            | Requires `response`? | Description                                                                                             |
# | ----------------------- | -------------------- | ------------------------------------------------------------------------------------------------------- |
# | **`context_precision`** | ❌ **No**             | Measures how relevant the retrieved `contexts` are to the `ground_truth` (question + expected answer)   |
# | **`faithfulness`**      | ✅ **Yes**            | Checks if the `response` is **faithful to the retrieved `contexts`** (i.e. did the LLM make things up?) |
# | **`answer_relevancy`**  | ✅ Yes                | Evaluates how well the `response` answers the `question`                                                |
# | **`context_recall`**    | ❌ No                 | Measures how much of the `ground_truth` is covered by the `contexts`                                    |
