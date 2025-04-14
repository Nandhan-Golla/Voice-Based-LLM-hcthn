from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import asyncio
from groq import AsyncGroq
import wikipedia
import requests
from datetime import datetime
import random
from cluster import (
    transcribe_audio, generate_response, get_weather, search_wikipedia,
    brainstorm_idea, track_goal, spark_creativity, process_command, process_memories
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024
ALLOWED_EXTENSIONS = {'txt'}


os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

conversation_context = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main interface."""
    return render_template('index.html')

@app.route('/api/process', methods=['POST'])
async def process():
    """Handle text or audio input."""
    global conversation_context
    try:
        if 'audio' in request.files:
            audio_file = request.files['audio']
            if audio_file:
            
                filename = secure_filename(audio_file.filename)
                temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio_file.save(temp_path)
                import numpy as np
                from scipy.io import wavfile
                sample_rate, audio_data = wavfile.read(temp_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data[:, 0]  # Mono
                audio_data = audio_data.astype(np.float32) / 32768.0
                transcription = await transcribe_audio(audio_data)
                os.unlink(temp_path)
                
                if transcription and not transcription.startswith("Whoops"):
                    response, conversation_context = await process_command(transcription, conversation_context)
                    return jsonify({'response': response, 'transcription': transcription})
                return jsonify({'error': 'Couldn’t transcribe audio'})
        
        elif 'text' in request.form:
            text = request.form['text']
            if text:
                response, conversation_context = await process_command(text, conversation_context)
                return jsonify({'response': response, 'transcription': text})
        
        return jsonify({'error': 'No input provided'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/api/upload_memories', methods=['POST'])
def upload_memories():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(temp_path)
            
            with open(temp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            response = process_memories(content)
            os.unlink(temp_path)
            return jsonify({'response': response})
        
        return jsonify({'error': 'Invalid file type'})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)