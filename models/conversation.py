import os
import json
import time
from datetime import datetime

class ConversationManager:
    def __init__(self, storage_dir="conversations"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
    
    def _get_user_file_path(self, user_id):
        """Generate file path for user conversations"""
        return os.path.join(self.storage_dir, f"user_{user_id}_conversation.json")
    
    def get_conversation(self, user_id):
        """Get conversation history for a user in a format suitable for the API"""
        file_path = self._get_user_file_path(user_id)
        
        if not os.path.exists(file_path):
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                conversation = json.load(f)
                
                # Filter to include only messages suitable for the DeepSeek API
                api_conversation = []
                for msg in conversation:
                    if msg.get("role") in ["user", "assistant", "system"]:
                        api_conversation.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
                
                return api_conversation
        except Exception as e:
            print(f"Error loading conversation for user {user_id}: {e}")
            return []
    
    def save_conversation(self, user_id, user_message, assistant_response):
        """Save user message and assistant response to conversation history"""
        conversation = self.get_conversation(user_id)
        
        # Add user message
        conversation.append({
            "role": "user",
            "content": user_message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Add assistant response
        conversation.append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save updated conversation
        file_path = self._get_user_file_path(user_id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving conversation for user {user_id}: {e}")
    
    def add_system_message(self, user_id, system_message, file_content=None):
        """Add system message to conversation history (e.g., for uploaded files)"""
        conversation = self.get_conversation(user_id)
        
        # Add system message
        conversation.append({
            "role": "system",
            "content": system_message,
            "timestamp": datetime.now().isoformat(),
            "file_content": file_content
        })
        
        # Save updated conversation
        file_path = self._get_user_file_path(user_id)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(conversation, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error saving system message for user {user_id}: {e}")