from flask import Flask, render_template_string
from flask_socketio import SocketIO
import webbrowser
import json

app = Flask(__name__)
socketio = SocketIO(app)

# HTML template with ElevenLabs widget
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>No-Code AI Builder</title>
  <style>
    html, body {
      margin: 0;
      padding: 0;
      height: 100%;
      width: 100%;
      overflow: hidden;
      background-color: #f5f5f5;
    }
    elevenlabs-convai {
      display: block;
      width: 100%;
      height: 100%;
    }
  </style>
</head>
<body>
  <elevenlabs-convai agent-id="YhnGsBX3Vp9EhmaNXW0i" aria-label="Voice-driven AI model builder"></elevenlabs-convai>
  <script src="https://elevenlabs.io/convai-widget/index.js" async type="text/javascript"></script>
  <script src="https://cdn.socket.io/4.5.0/socket.io.min.js"></script>
  <script>
    const socket = io();
    // Mock widget API interaction (replace with actual ElevenLabs API if available)
    document.addEventListener('DOMContentLoaded', () => {
      const widget = document.querySelector('elevenlabs-convai');
      // Simulate capturing voice input (actual implementation depends on widget API)
      setInterval(() => {
        const mockInput = { text: "Build a model to predict sales" }; // Replace with widget's output
        socket.emit('voice_command', mockInput);
      }, 5000);
    });
  </script>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(html_template)

@socketio.on('voice_command')
def handle_voice_command(data):
    command = data.get('text', '')
    print(f"Received command: {command}")
    # Mock AI pipeline (replace with real model-building logic)
    response = {"status": "Processing", "message": f"Building model based on: {command}"}
    socketio.emit('ai_response', response)

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:5000")
    socketio.run(app, debug=True)