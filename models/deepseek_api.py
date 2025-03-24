# models/deepseek_api.py
import os
import json
from openai import OpenAI

class DeepSeekAPI:
    def __init__(self, api_key=None):
        # Make sure we have a valid API key
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DeepSeek API key is missing. Please set DEEPSEEK_API_KEY environment variable or pass api_key to DeepSeekAPI.")
        
        self.base_url = os.getenv('DEEPSEEK_API_URL', 'https://api.deepseek.com')
        self.model_name = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
        self.admin_instruction = ""
        self.admin_instruction_file = "admin_instruction.txt"
        
        # Initialize OpenAI client with explicit API key
        try:
            # Create client without proxies argument which seems to be causing issues
            client_kwargs = {
                "api_key": self.api_key,
                "base_url": self.base_url
            }
            
            # Initialize the client with compatible arguments
            self.client = OpenAI(**client_kwargs)
            print(f"Successfully initialized DeepSeek API client with base URL: {self.base_url}")
        except Exception as e:
            print(f"Error initializing DeepSeek API client: {e}")
            raise
        
        # Load admin instruction if exists
        self._load_admin_instruction()
    
    def _load_admin_instruction(self):
        """Load admin instruction from file if exists"""
        try:
            if os.path.exists(self.admin_instruction_file):
                with open(self.admin_instruction_file, 'r', encoding='utf-8') as f:
                    self.admin_instruction = f.read().strip()
        except Exception as e:
            print(f"Error loading admin instruction: {e}")
    
    def set_admin_instruction(self, instruction):
        """Set and save admin instruction that will be prepended to all user queries"""
        self.admin_instruction = instruction.strip()
        
        # Save to file
        try:
            with open(self.admin_instruction_file, 'w', encoding='utf-8') as f:
                f.write(self.admin_instruction)
        except Exception as e:
            print(f"Error saving admin instruction: {e}")
    
    def generate_response(self, message, conversation_history=None):
        """Generate response from DeepSeek V3 model using the OpenAI compatible API"""
        if conversation_history is None:
            conversation_history = []
        
        # Prepare messages including admin instruction as system message
        messages = []
        
        # Add admin instruction as system message if available
        if self.admin_instruction:
            messages.append({
                "role": "system",
                "content": self.admin_instruction
            })
        elif not any(msg.get("role") == "system" for msg in conversation_history):
            # Add default system message if no system message in history
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant."
            })
        
        # Filter conversation history to include only the necessary fields
        # and ensure proper format for the API
        for msg in conversation_history:
            if msg.get("role") in ["user", "assistant", "system"]:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Add current user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        try:
            # Print debug info
            print(f"Calling DeepSeek API with model: {self.model_name}")
            print(f"Message count: {len(messages)}")
            
            # Call the DeepSeek API using the OpenAI client
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7,
                stream=False
            )
            
            # Extract and return the response content
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}")
            return f"Sorry, I encountered an error while processing your request: {str(e)}"