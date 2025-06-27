import pandas as pd
import os
from langchain_community.document_loaders.csv_loader import CSVLoader


# Set the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create a document loader for fifa_countries_audience.csv
loader = CSVLoader("01-dev-llm-app/03/fifa_countries_audience.csv")

# Load the document
data = loader.load() 
# print(data[0])


# Explore the loaded data
print(f"Number of rows loaded: {len(data)}")
print(f"First document:")
print(data[0])
print(f"\nFirst document metadata: {data[0].metadata}")

# Look at a few more rows to understand the pattern
print(f"\nFirst few countries:")
for i in range(min(3, len(data))):
    # Extract just the country name from page_content
    content = data[i].page_content
    country_line = content.split('\n')[0]  # First line should be country
    print(f"Row {i}: {country_line}")


# More advanced configuration
# loader = CSVLoader(
#     file_path="fifa_countries_audience.csv",
#     csv_args={
#         'delimiter': ',',           # Specify delimiter
#         'quotechar': '"',          # Specify quote character
#         'fieldnames': ['country', 'confederation', 'population_share']  # Specify column names
#     }
# )