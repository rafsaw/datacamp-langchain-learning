import pandas as pd
import os
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter, Language , TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings 
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import shutil
from langchain_community.document_loaders import PythonLoader
from langchain_openai  import OpenAIEmbeddings 
from langchain_experimental.text_splitter  import SemanticChunker 
import tiktoken 

openai_api_key = os.getenv("OPENAI_API_KEY")

# Get the encoding for gpt-4o-mini
encoding = tiktoken.encoding_for_model('gpt-4o-mini')

# Create a token text splitter
token_splitter = TokenTextSplitter(encoding_name=encoding.name, chunk_size=100, chunk_overlap=10)

# Load the PDF
loader = PyPDFLoader("02-RAG-LangChain/AuditSummaryDraft-ENG_CMPY-157914_ACTY-2022-532608.pdf")
document = loader.load()

# Split the PDF into chunks
chunks = token_splitter.split_documents(document) 

# for i, chunk in enumerate(chunks[:3]):
#     print(f"Chunk {i+1}:\nNo. tokens: {len(encoding.encode(chunk.page_content))}\n{chunk}\n")

# Instantiate an OpenAI embeddings model
embedding_model = OpenAIEmbeddings(api_key=openai_api_key, model='text-embedding-3-small')

# Create the semantic text splitter with desired parameters
semantic_splitter = SemanticChunker(
    embeddings=embedding_model, breakpoint_threshold_type="gradient", breakpoint_threshold_amount=0.8
)

# Split the document
chunks = semantic_splitter.split_documents(document) 
print(chunks[0])

# Save chunks to text file with clear identification
def save_chunks_to_file(chunks, output_filename="document_chunks.txt"):
    """
    Save chunks to a text file with clear identification of each chunk.
    
    Args:
        chunks: List of document chunks
        output_filename: Name of the output file
    """
    # Get the current working directory and create full path
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, output_filename)
    
    print(f"Saving file to: {full_path}")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("DOCUMENT CHUNKS ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Total number of chunks: {len(chunks)}\n")
        f.write(f"Source document: AuditSummaryDraft-ENG_CMPY-157914_ACTY-2022-532608.pdf\n")
        f.write(f"Splitter type: SemanticChunker\n")
        f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Saved to: {full_path}\n\n")
        
        for i, chunk in enumerate(chunks, 1):
            f.write("-" * 60 + "\n")
            f.write(f"CHUNK #{i:03d}\n")
            f.write("-" * 60 + "\n")
            
            # Write chunk metadata
            f.write(f"Chunk ID: {i}\n")
            f.write(f"Page: {chunk.metadata.get('page', 'N/A')}\n")
            f.write(f"Source: {chunk.metadata.get('source', 'N/A')}\n")
            
            # Calculate token count if encoding is available
            try:
                token_count = len(encoding.encode(chunk.page_content))
                f.write(f"Token count: {token_count}\n")
            except:
                f.write(f"Character count: {len(chunk.page_content)}\n")
            
            f.write(f"Content length: {len(chunk.page_content)} characters\n")
            f.write("\n")
            
            # Write the actual content
            f.write("CONTENT:\n")
            f.write(chunk.page_content)
            f.write("\n\n")
            
            # Add a separator for readability
            f.write("=" * 60 + "\n\n")
    
    print(f"✓ Chunks saved to: {output_filename}")
    print(f"✓ Full path: {full_path}")
    print(f"✓ Total chunks written: {len(chunks)}")
    
    # Verify file was created
    if os.path.exists(output_filename):
        file_size = os.path.getsize(output_filename)
        print(f"✓ File size: {file_size:,} bytes")
    else:
        print("❌ Error: File was not created!")

# Save both token-based and semantic chunks for comparison
print("\n" + "="*50)
print("SAVING TOKEN-BASED CHUNKS")
print("="*50)
token_chunks = token_splitter.split_documents(document)
save_chunks_to_file(token_chunks, "token_chunks.txt")

print("\n" + "="*50)
print("SAVING SEMANTIC CHUNKS")
print("="*50)
save_chunks_to_file(chunks, "semantic_chunks.txt")

# Optional: Save a summary file
def save_chunks_summary(chunks, output_filename="chunks_summary.txt"):
    """
    Save a summary of all chunks with basic statistics.
    """
    import os
    current_dir = os.getcwd()
    full_path = os.path.join(current_dir, output_filename)
    
    print(f"Saving summary to: {full_path}")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("CHUNKS SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Total chunks: {len(chunks)}\n")
        
        # Calculate statistics
        content_lengths = [len(chunk.page_content) for chunk in chunks]
        token_counts = [len(encoding.encode(chunk.page_content)) for chunk in chunks]
        
        f.write(f"Average content length: {sum(content_lengths) / len(content_lengths):.1f} characters\n")
        f.write(f"Min content length: {min(content_lengths)} characters\n")
        f.write(f"Max content length: {max(content_lengths)} characters\n")
        f.write(f"Average token count: {sum(token_counts) / len(token_counts):.1f} tokens\n")
        f.write(f"Min token count: {min(token_counts)} tokens\n")
        f.write(f"Max token count: {max(token_counts)} tokens\n\n")
        
        f.write("CHUNK DETAILS:\n")
        f.write("-" * 30 + "\n")
        
        for i, chunk in enumerate(chunks, 1):
            token_count = len(encoding.encode(chunk.page_content))
            f.write(f"Chunk {i:03d}: {len(chunk.page_content)} chars, {token_count} tokens, Page {chunk.metadata.get('page', 'N/A')}\n")

print("\n" + "="*50)
print("SAVING SUMMARY FILES")
print("="*50)
# Save summary
save_chunks_summary(chunks, "semantic_chunks_summary.txt")
save_chunks_summary(token_chunks, "token_chunks_summary.txt")

# Final summary of all created files
print("\n" + "="*50)
print("FINAL SUMMARY - ALL CREATED FILES")
print("="*50)
import os
current_dir = os.getcwd()
created_files = ["token_chunks.txt", "semantic_chunks.txt", "token_chunks_summary.txt", "semantic_chunks_summary.txt"]

for filename in created_files:
    full_path = os.path.join(current_dir, filename)
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        print(f"✓ {filename}: {file_size:,} bytes")
        print(f"  Location: {full_path}")
    else:
        print(f"❌ {filename}: NOT FOUND")

print(f"\nWorking directory: {current_dir}")

print(f"Current working directory: {os.getcwd()}")
