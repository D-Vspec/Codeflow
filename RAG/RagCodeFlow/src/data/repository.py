# src/data/repository.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

def read_repository_files(repo_path, ignore_dirs):
    """
    Read all relevant files from a repository
    
    Args:
        repo_path (str): Path to the repository
        ignore_dirs (list): List of directories to ignore
        
    Returns:
        dict: Dictionary of file paths to file contents
    """
    file_contents = {}
    print("\n--- PROCESSING FILES ---")
    file_count = 0
    
    for root, dirs, files in os.walk(repo_path):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not any(ignored in root for ignored in ignore_dirs)]
        
        for file in files:
            # Focus on code and documentation files, skip build artifacts
            if file.endswith((".py", ".js", ".ts", ".md", ".tsx", ".jsx", ".json")) and not any(build_dir in root for build_dir in [".cxx", "build"]):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                
                try:
                    # Print each file being processed
                    file_count += 1
                    print(f"{file_count}. {rel_path}")
                    
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        file_contents[file_path] = f.read()
                except Exception as e:
                    print(f"   ERROR reading file {rel_path}: {e}")
    
    print(f"\nTotal files processed: {file_count}")
    return file_contents

def chunk_documents(repo_data):
    """
    Split documents into manageable chunks
    
    Args:
        repo_data (dict): Dictionary of file paths to file contents
        
    Returns:
        list: List of dictionaries containing path and content chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    for path, content in repo_data.items():
        for chunk in text_splitter.split_text(content):
            chunks.append({"path": path, "content": chunk})
    return chunks
