import hashlib
import os
import json
from werkzeug.security import generate_password_hash, check_password_hash

class Auth:
    def __init__(self, users_file="users.json"):
        self.users_file = users_file
        self._ensure_users_file()
    
    def _ensure_users_file(self):
        """Create users file with default admin if it doesn't exist"""
        if not os.path.exists(self.users_file):
            default_admin = {
                "id": "1",
                "username": "admin",
                "password": generate_password_hash("admin"),
                "is_admin": True
            }
            
            default_user = {
                "id": "2",
                "username": "user",
                "password": generate_password_hash("user"),
                "is_admin": False
            }
            
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump([default_admin, default_user], f, indent=2)
    
    def get_users(self):
        """Get all users"""
        try:
            with open(self.users_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading users: {e}")
            return []
    
    def login(self, username, password):
        """Authenticate user"""
        users = self.get_users()
        
        for user in users:
            if user['username'] == username and check_password_hash(user['password'], password):
                # Return user without password
                return {k: v for k, v in user.items() if k != 'password'}
        
        return None
    
    def add_user(self, username, password, is_admin=False):
        """Add new user"""
        users = self.get_users()
        
        # Check if username already exists
        if any(user['username'] == username for user in users):
            return False
        
        # Generate new user ID
        user_id = str(max(int(user['id']) for user in users) + 1) if users else "1"
        
        new_user = {
            "id": user_id,
            "username": username,
            "password": generate_password_hash(password),
            "is_admin": is_admin
        }
        
        users.append(new_user)
        
        # Save updated users
        try:
            with open(self.users_file, 'w', encoding='utf-8') as f:
                json.dump(users, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving user: {e}")
            return False