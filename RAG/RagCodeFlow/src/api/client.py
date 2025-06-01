# src/api/client.py
from openai import OpenAI

def generate_api_response(context, query, api_key):
    """
    Generate a response using DeepSeek API with OpenAI SDK format
    
    Args:
        context (str): The relevant code snippets from the repository
        query (str): The user's query about the codebase
        api_key (str): API key for DeepSeek
        
    Returns:
        str: The generated response analyzing the code
    """
    print("Calling DeepSeek API using OpenAI SDK format...")
    
    try:
        # Initialize client with DeepSeek base URL
        client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
        # Prepare system and user messages
        system_message = "You are a helpful code assistant. Analyze the provided code and answer questions about it."
        user_message = f"""
Here are relevant chunks from the codebase:

{context}

Based on these code excerpts, please answer the following question:
{query}
"""
        
        # Make API call
        response = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        # Extract and return the response content
        return response.choices[0].message.content
    
    except Exception as e:
        print(f"API request error: {str(e)}")
        return f"Error calling DeepSeek API: {str(e)}"
