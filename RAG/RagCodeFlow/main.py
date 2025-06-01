from fastapi import FastAPI, HTTPException
import json
import os
import sys
from src.api.client import generate_api_response
from src.data.repository import read_repository_files, chunk_documents
from src.embeddings.vectorstore import create_embeddings, build_faiss_index
from src.models.inference import process_query

app = FastAPI()

# Configuration
BASE_REPO_PATH = os.path.join(sys.base_prefix, "repo")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
API_KEY = "sk-806fee1db5204464b271879257cd769e"  # Replace with your actual DeepSeek API key
QUERY = """
What is the purpose of the code?
Is there code that serves the same purpose written in multiple places at once?
Are there any logical errors?
Are there any syntax errors?
What improvements can be made?

Give the output in this format: 
{
        "Summary": "The code is part of a React Native application for managing entities like bookings, partners, users, and boats. It includes pagination, CRUD operations, and UI components like charts, modals, and carousels.",
        "Redundancy": [
            {
                "Description": "Pagination logic (`handlePrevPage`, `handleNextPage`, `handleRowsPerPageChange`) is repeated in multiple files.",
                "Files": ["bookings.tsx", "partnerScreen.tsx", "users.tsx"]
            },
            {
                "Description": "Edit handlers for different entities (`handleEditPartner`, `handleEditUser`) follow the same pattern.",
                "Files": ["partnerScreen.tsx", "users.tsx"]
            }
        ],
        "LogicalErrors": [
            {
                "Description": "Pagination does not handle edge cases where `totalPages` is 0, potentially causing UI inconsistencies.",
                "File": "bookings.tsx"
            },
            {
                "Description": "State resets are incomplete in `handleAddBooking`, possibly leading to stale form data.",
                "File": "bookings.tsx"
            }
        ],
        "SyntaxErrors": [
            {
                "Description": "Possible missing closing tags in JSX files.",
                "Files": ["carousel.tsx", "expenseModal.tsx"]
            }
        ],
        "Improvements": [
            {
                "Description": "Extract pagination logic into a reusable hook.",
                "Suggestion": "Create `usePagination` to handle page state and navigation."
            },
            {
                "Description": "Add safeguards for invalid `totalPages`.",
                "Suggestion": "Ensure `totalPages = Math.max(1, totalPages)` to prevent pagination errors."
            },
            {
                "Description": "Generalize edit handlers into a single function with entity type as a parameter.",
                "Suggestion": "Refactor `handleEditPartner` and `handleEditUser` into a shared function."
            },
            {
                "Description": "Update outdated dependencies.",
                "Suggestion": "Upgrade `chalk@4.0.0` and `debug@4.3.4` to newer versions."
            },
            {
                "Description": "Improve TypeScript typing.",
                "Suggestion": "Define stricter types for entities like `Booking` and `User` to reduce `any` usage."
            }
        ]
    }
Do not Hallucinate
"""

IGNORE_DIRS = ["venv", "node_modules", ".git", "__pycache__", ".cxx", "build", "android/app/.cxx", "android/app/build"]

def validate_response(response_json):
    required_keys = {"Summary", "Redundancy", "LogicalErrors", "SyntaxErrors", "Improvements"}
    if not isinstance(response_json, dict) or not required_keys.issubset(response_json.keys()):
        raise ValueError("Invalid response format")
    return response_json

@app.get("/analyze/{repo_name}")
def analyze_repository(repo_name: str):
    print(BASE_REPO_PATH)
    repo_path = os.path.join(BASE_REPO_PATH, repo_name)
    
    if not os.path.exists(repo_path):
        raise HTTPException(status_code=404, detail=f"Repository not found on {repo_path}")
    
    repo_data = read_repository_files(repo_path, IGNORE_DIRS)
    if not repo_data:
        raise HTTPException(status_code=400, detail="No valid files found in the repository")
    
    chunks = chunk_documents(repo_data)
    vector_data = create_embeddings(chunks, EMBEDDING_MODEL)
    index, vector_matrix = build_faiss_index(vector_data)
    
    query_vector, retrieved_chunks = process_query(QUERY, EMBEDDING_MODEL, index, vector_data)
    context = "\n\n".join(retrieved_chunks)
    
    response = generate_api_response(context, QUERY, API_KEY)
    
    try:
        response_json = json.loads(response)
        validated_response = validate_response(response_json)
        print(validated_response)
        return validated_response
    except (json.JSONDecodeError, ValueError) as e:
        print(response)
        raise HTTPException(status_code=500, detail=f"Invalid response format: {e}")
