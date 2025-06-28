from ragas.metrics import context_precision
from ragas import evaluate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from datasets import Dataset

# Set up API keys and LLM
openai_api_key = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)

# In a typical RAG pipeline, the steps are:
    # User asks a question → "question"
    # Your retriever searches your document database
    # The retriever returns relevant text chunks → these go into "contexts"
    # The LLM generates an answer → (optional for some metrics)
    # RAGAS evaluates how good the retrieved "contexts" were (and optionally the final answer)


# contexts are the chunks returned by search (retriever).

# Create evaluation dataset
eval_dataset = Dataset.from_dict({
    "question": ["What are knowledge-intensive tasks?"],
    "contexts": [["Knowledge-intensive tasks are tasks that require specialized knowledge to complete."]],
    "ground_truth": ["Knowledge-intensive tasks are those that rely on deep expertise or domain-specific knowledge."]
})

# Run evaluation
results = evaluate(
    eval_dataset,
    metrics=[context_precision],
    llm=llm,
    embeddings=embeddings
)

print("Evaluation Context Precision Results:")
print(results)
