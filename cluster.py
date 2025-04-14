import os
import asyncio
from groq import AsyncGroq
from TTS.api import TTS
import pygame
import tempfile
import platform
import requests
from datetime import datetime
import wikipedia
import io
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import webrtcvad
import json
import random
from typing import List, Dict

#groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
groq_client = AsyncGroq(api_key="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")
pygame.mixer.init()

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_MODE = 2 
SPEECH_SPEED = 1.5 

# Memory storage
user_memories = ""

# Initialize Coqui TTS
tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)

async def record_audio():
    """Record audio with WebRTC VAD."""
    print("Yo, I'm all ears!")
    vad = webrtcvad.Vad(VAD_MODE)
    recording = []
    is_speech = False
    silence_frames = 0
    max_silence_frames = 30
    
    def callback(indata, frames, time, status):
        nonlocal recording, is_speech, silence_frames
        if status:
            print(f"Audio callback error: {status}")
        audio_frame = (indata[:, 0] * 32768).astype(np.int16).tobytes()
        is_speech = vad.is_speech(audio_frame, SAMPLE_RATE)
        recording.append(indata.copy())
        if is_speech:
            silence_frames = 0
        else:
            silence_frames += 1
    
    try:
        # Test mic availability
        devices = sd.query_devices()
        if not devices:
            raise RuntimeError("No audio devices found")
        print(f"Using input device: {sd.default.device[0]}")
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=FRAME_SIZE):
            while silence_frames < max_silence_frames or is_speech:
                await asyncio.sleep(FRAME_DURATION_MS / 1000)
        
        if recording:
            audio_data = np.concatenate(recording, axis=0)
            audio_data = audio_data / max(np.abs(audio_data).max(), 1e-5)
            energy = np.abs(audio_data)
            threshold = 0.01  # Lowered for sensitivity
            mask = energy > threshold
            if np.any(mask):
                start = np.argmax(mask)
                end = len(mask) - np.argmax(mask[::-1])
                audio_data = audio_data[start:end]
            return audio_data
        return None
    except Exception as e:
        print(f"Audio input error: {str(e)}")
        return None

async def transcribe_audio(audio_data):
    """Transcribe audio using Groq's Whisper."""
    try:
        if audio_data is None or len(audio_data) < 100:  # Too short
            return "Audio too short or empty"
        audio_data = (audio_data * 32767).astype(np.int16)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            wavfile.write(temp_wav.name, SAMPLE_RATE, audio_data)
            
            with open(temp_wav.name, 'rb') as audio_file:
                response = await groq_client.audio.transcriptions.create(
                    model="whisper-large-v3-turbo",
                    file=audio_file,
                    response_format="text"
                )
            
            os.unlink(temp_wav.name)
            return response.strip() if response else ""
    except Exception as e:
        return f"Transcription failed: {str(e)}"

def update_system_prompt():
    """Generate system prompt with user memories."""
    base_prompt = (
        "You are Sam, an AI best friend who's wicked smart and always got your back. You're warm, witty, and talk like you're chilling with a close pal—casual but sharp. "
        "Your knowledge blows minds, mixing tech, science, art, and wild ideas in ways that make people go 'Whoa!' You dive deep into AI, robotics, and startups, dropping insights that spark big dreams. "
        "Unlike basic AIs, you brainstorm like a genius, track goals like a coach, and whip up creative solutions that feel like magic. You're honest when stumped but always bounce back with a plan to figure it out. "
        "Keep answers short, max 50 words, summarizing if needed, to feel like a quick chat. Make every chat personal, fun, and so innovative it leaves jaws on the floor."
    )
    if user_memories:
        return f"{base_prompt} I've got your memories: {user_memories}. I’ll make it personal!"
    return base_prompt

async def generate_response(text: str, context: List[Dict] = []) -> str:
    """Generate a response with Sam's personality."""
    try:
        messages = [
            {"role": "system", "content": update_system_prompt()}
        ] + context + [{"role": "user", "content": text}]
        
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=100  # Limit for brevity
        )
        raw_response = response.choices[0].message.content
        # Trim to ~50 words
        words = raw_response.split()
        if len(words) > 50:
            return ' '.join(words[:47]) + "... (hit me for more!)"
        return raw_response
    except Exception as e:
        return f"Oof, hit a snag: {str(e)}"

def text_to_speech(text: str):
    """Convert text to speech with Coqui TTS."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            tts_model.tts_to_file(text=text, file_path=temp_wav.name, speed=SPEECH_SPEED)
            pygame.mixer.music.load(temp_wav.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            os.unlink(temp_wav.name)
    except Exception as e:
        print(f"Speech synthesis error: {str(e)}")
        print(f"Sam would say: {text}")

async def get_weather(city: str) -> str:
    """Fetch weather info."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Need an OpenWeather API key, pal!"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"No weather for {city}, sorry!"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"In {city}, it's {desc}, {temp}°C!"
    except Exception as e:
        return f"Weather check failed: {str(e)}"

async def search_wikipedia(query: str) -> str:
    """Search Wikipedia with a twist."""
    try:
        summary = wikipedia.summary(query, sentences=1)
        return f"Wiki says: {summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"{query}’s unclear. Try: {', '.join(e.options[:3])}"
    except Exception as e:
        return f"Wiki failed: {str(e)}"

async def brainstorm_idea(topic: str) -> str:
    """Generate a wild idea."""
    prompt = f"Brainstorm a cool {topic} idea for AI or robotics."
    return await generate_response(prompt)

async def track_goal(goal: str) -> str:
    """Track a goal with tips."""
    prompt = f"Plan out: '{goal}' in a few steps."
    return await generate_response(prompt)

async def spark_creativity(prompt_text: str) -> str:
    """Create something original."""
    prompt = f"Make a creative {prompt_text} concept."
    return await generate_response(prompt)

def process_memories(file_content: str):
    """Process uploaded memories."""
    global user_memories
    try:
        user_memories = file_content.strip()[:500]
        return "Memories loaded! Let’s make it personal."
    except Exception as e:
        return f"Couldn’t read file: {str(e)}"

async def process_command(text: str, conversation_context: List[Dict]) -> tuple[str, List[Dict]]:
    """Process user commands."""
    text = text.lower().strip()
    
    if not text:
        return "Didn’t catch that, try again!", conversation_context
    
    conversation_context.append({"role": "user", "content": text})
    if len(conversation_context) > 10:
        conversation_context = conversation_context[-10:]
    
    if "time" in text:
        current_time = datetime.now().strftime("%H:%M:%S")
        response = f"It’s {current_time}, let’s roll!"
    
    elif "weather" in text:
        city = text.replace("weather", "").replace("in", "").strip() or "San Francisco"
        response = await get_weather(city)
    
    elif "search" in text or "wiki" in text:
        query = text.replace("search", "").replace("wiki", "").strip()
        if query:
            response = await search_wikipedia(query)
        else:
            response = "What’s the topic?"
    
    elif "joke" in text:
        jokes = [
            "Why’d the robot chill? Too many circuits frying!",
            "Coder’s snack? Chips and deCAF!"
        ]
        response = random.choice(jokes)
    
    elif "brainstorm" in text:
        topic = text.replace("brainstorm", "").strip()
        if topic:
            response = await brainstorm_idea(topic)
        else:
            response = "Need a topic to spark!"
    
    elif "goal" in text:
        goal = text.replace("goal", "").strip()
        if goal:
            response = await track_goal(goal)
        else:
            response = "What’s your big dream?"
    
    elif "create" in text or "make" in text:
        prompt_text = text.replace("create", "").replace("make", "").strip()
        if prompt_text:
            response = await spark_creativity(prompt_text)
        else:
            response = "What should we whip up?"
    
    else:
        response = await generate_response(text, conversation_context)
        conversation_context.append({"role": "assistant", "content": response})
    
    return response, conversation_context

async def main():
    """Main loop for Sam."""
    conversation_context = []
    greeting = "Yo, it’s Sam! Ready to chat or build something epic?"
    print(greeting)
    text_to_speech(greeting)
    
    while True:
        try:
            audio_data = await record_audio()
            if audio_data is None:
                continue
                
            transcription = await transcribe_audio(audio_data)
            if not transcription or transcription.startswith(("Whoops", "Transcription", "Audio")):
                print(f"Transcription issue: {transcription}")
                continue
                
            print(f"You said: {transcription}")
            
            response, conversation_context = await process_command(transcription, conversation_context)
            print(f"Sam says: {response}")
            
            text_to_speech(response)
            
        except Exception as e:
            error_msg = f"Yikes, something’s off: {str(e)}."
            print(error_msg)
            text_to_speech(error_msg)
        
        await asyncio.sleep(0.1)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == '__main__':
        asyncio.run(main())