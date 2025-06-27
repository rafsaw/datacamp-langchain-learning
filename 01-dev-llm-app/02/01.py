import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser



# Load your OpenAI key from .env
load_dotenv()

# Create an OpenAI chat LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))

# Create a prompt template that takes an input activity
learning_prompt = PromptTemplate(
    input_variables=["activity"],
    template="I want to learn how to {activity}. Can you suggest how I can learn this step-by-step in Polish?"
)

# Create a prompt template that places a time constraint on the output
time_prompt = PromptTemplate(
    input_variables=["learning_plan"],
    template="I only have one week. Can you create a plan to help me hit this goal: {learning_plan} in Polish."
)

# Invoke the learning_prompt with an activity
print(learning_prompt.invoke({"activity": "play golf"}))

# Complete the sequential chain with LCEL
seq_chain = ({"learning_plan": learning_prompt | llm | StrOutputParser()}
    | time_prompt
    | llm
    | StrOutputParser())


# Call the chain
print(seq_chain.invoke({"activity": "travel to south of Italy for 10 days"}))