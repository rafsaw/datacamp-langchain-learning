import pandas as pd
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter

quote = 'Words are flowing out like endless rain into a paper cup,\nthey slither while they pass,\nthey slip away across the universe.'
chunk_size = 24
chunk_overlap = 10

# RecursiveCharacterTextSplitter tries multiple separators
rc_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],  # Tries in order: paragraphs, lines, words, characters
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

docs_recursive = rc_splitter.split_text(quote)
print("RecursiveCharacterTextSplitter:", docs_recursive)
print("Lengths:", [len(doc) for doc in docs_recursive])