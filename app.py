from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
import os
from waitress import serve
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

from models.deepseek_api import DeepSeekAPI
from models.conversation import ConversationManager
from models.auth import Auth
from tts_service import generate_voice

# 确保必要的目录存在
os.makedirs('voice', exist_ok=True)

# Load environment variables
load_dotenv('environment.env')

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')
app.config['UPLOAD_FOLDER'] = os.getenv('UPLOAD_FOLDER', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize services
auth = Auth()
deepseek_api = DeepSeekAPI(api_key=os.getenv('DEEPSEEK_API_KEY'))
conversation_manager = ConversationManager(storage_dir='conversations')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'csv'}
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('chat.html', is_admin=session.get('is_admin', False))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        user = auth.login(username, password)
        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            session['is_admin'] = user['is_admin']
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/chat', methods=['POST'])
def chat():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    message = request.form.get('message', '')
    
    # Get conversation history
    conversation = conversation_manager.get_conversation(user_id)
    
    # Call DeepSeek API
    response = deepseek_api.generate_response(message, conversation)
    
    # Save conversation
    conversation_manager.save_conversation(user_id, message, response)
    
    # 生成语音
    try:
        voice_file = generate_voice(response)
        if voice_file and os.path.exists(voice_file):
            voice_relative_path = "/voice/" + os.path.basename(voice_file)
        else:
            print("警告：语音文件未成功生成或不存在")
            voice_relative_path = None
    except Exception as e:
        print(f"生成语音时出错: {e}")
        voice_relative_path = None
    
    return {
        'response': response,
        'voice_file': voice_relative_path
    }

@app.route('/admin/instruction', methods=['GET', 'POST'])
def admin_instruction():
    if not session.get('is_admin', False):
        return redirect(url_for('home'))
    
    if request.method == 'POST':
        instruction = request.form.get('instruction', '')
        if instruction:
            deepseek_api.set_admin_instruction(instruction)
            flash('Admin instruction has been set successfully!')
        
    return render_template('admin_instruction.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if 'file' not in request.files:
        return {'error': 'No file part'}, 400
    
    file = request.files['file']
    if file.filename == '':
        return {'error': 'No selected file'}, 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read the file content
        with open(filepath, 'r', encoding='utf-8') as f:
            file_content = f.read()
        
        # Add file content to conversation
        user_id = session['user_id']
        system_message = f"[System] User uploaded file: {filename}. File content: {file_content[:1000]}..."
        conversation_manager.add_system_message(user_id, system_message, file_content)
        
        return {'success': True, 'message': f'File {filename} has been uploaded and processed'}
    
    return {'error': 'File type not allowed'}, 400

# 添加静态文件服务路由，使前端能够访问voice文件夹中的音频文件
@app.route('/voice/<path:filename>')
def serve_voice(filename):
    return send_from_directory('voice', filename)

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8080))
    print(f"启动服务器在 0.0.0.0:{port}")
    serve(app, host='0.0.0.0', port=port)