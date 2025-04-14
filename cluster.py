import os
import asyncio
from groq import AsyncGroq
import requests
import pygame
import tempfile
import platform
import wikipedia
import io
from scipy.io import wavfile
import numpy as np
import sounddevice as sd
import webrtcvad
import json
import random
from datetime import datetime
from typing import List, Dict

#groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
groq_client = AsyncGroq(api_key="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")
pygame.mixer.init()
SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_MODE = 1 
SPEECH_SPEED = 1.5
#ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_API_KEY = "sk_bcbd83473a087c70276e72189cfe64c476cd8900bc9b3735"
user_memories = ""

async def record_audio():

    print("Yo, I'm all ears! Speak clearly...")
    vad = webrtcvad.Vad(VAD_MODE)
    recording = []
    is_speech = False
    silence_frames = 0
    max_silence_frames = 50  
    
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
        devices = sd.query_devices()
        if not devices:
            raise RuntimeError("No audio input devices found")
        print(f"Using input device: {sd.default.device[0]} ({devices[sd.default.device[0]]['name']})")
        
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=FRAME_SIZE):
            while silence_frames < max_silence_frames or is_speech:
                await asyncio.sleep(FRAME_DURATION_MS / 1000)
        
        if recording:
            audio_data = np.concatenate(recording, axis=0)
            if np.max(np.abs(audio_data)) < 0.01:
                print("Audio too quiet, discarding")
                return None
            audio_data = audio_data / max(np.abs(audio_data).max(), 1e-5)
            energy = np.abs(audio_data)
            threshold = 0.01
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
    try:
        if audio_data is None or len(audio_data) < 200:
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
            if not response or response.strip() == "":
                return "No speech detected"
            return response.strip()
    except Exception as e:
        return f"Transcription error: {str(e)}"

def update_system_prompt():
    base_prompt = (
        "You are Sam, an AI best friend—super smart, warm, and witty. Talk like you're chilling with a pal, keeping it casual but sharp. "
        "Drop mind-blowing insights on AI, robotics, startups, or anything cool, but keep answers short—50 words max, summarize if needed. "
        "Be honest if stumped, then find a way forward. Make chats fun, personal, and innovative to wow everyone."
    )
    if user_memories:
        return f"{base_prompt} Got your memories: {user_memories}. Let’s make it personal!"
    return base_prompt

async def generate_response(text: str, context: List[Dict] = []) -> str:
    try:
        messages = [
            {"role": "system", "content": update_system_prompt()}
        ] + context + [{"role": "user", "content": text}]
        
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=80
        )
        raw_response = response.choices[0].message.content
        words = raw_response.split()
        if len(words) > 50:
            return ' '.join(words[:47]) + "... (ask for more!)"
        return raw_response
    except Exception as e:
        return f"Hit a snag: {str(e)}"

def text_to_speech(text: str):
    if not ELEVENLABS_API_KEY:
        print(f"No ElevenLabs API key. Sam would say: {text}")
        return
    
    try:
        url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
                "speed": SPEECH_SPEED
            }
        }
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
            temp_mp3.write(response.content)
            temp_mp3.flush()
            pygame.mixer.music.load(temp_mp3.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            os.unlink(temp_mp3.name)
    except Exception as e:
        print(f"Speech synthesis error: {str(e)}")
        print(f"Sam would say: {text}")

async def get_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Need an OpenWeather API key!"
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"No weather for {city}!"
        temp = data["main"]["temp"]
        desc = data["weather"][0]["description"]
        return f"{city}: {desc}, {temp}°C!"
    except Exception as e:
        return f"Weather failed: {str(e)}"

async def search_wikipedia(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=1)
        return f"Wiki: {summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Clarify {query}: {', '.join(e.options[:3])}"
    except Exception as e:
        return f"Wiki error: {str(e)}"

async def brainstorm_idea(topic: str) -> str:
    prompt = f"Spark a {topic} idea for AI or robotics."
    return await generate_response(prompt)

async def track_goal(goal: str) -> str:
    prompt = f"Plan '{goal}' in a few steps."
    return await generate_response(prompt)

async def spark_creativity(prompt_text: str) -> str:
    prompt = f"Craft a {prompt_text} concept."
    return await generate_response(prompt)

def process_memories(file_content: str):
    global user_memories
    try:
        user_memories = file_content.strip()[:500]
        return "Memories loaded! Ready to vibe."
    except Exception as e:
        return f"Memory error: {str(e)}"

async def process_command(text: str, conversation_context: List[Dict]) -> tuple[str, List[Dict]]:
    text = text.lower().strip()
    
    if not text:
        return "Didn’t hear ya, try again!", conversation_context
    
    conversation_context.append({"role": "user", "content": text})
    if len(conversation_context) > 10:
        conversation_context = conversation_context[-10:]
    
    if "time" in text:
        current_time = datetime.now().strftime("%H:%M:%S")
        response = f"It’s {current_time}, let’s go!"
    
    elif "weather" in text:
        city = text.replace("weather", "").replace("in", "").strip() or "San Francisco"
        response = await get_weather(city)
    
    elif "search" in text or "wiki" in text:
        query = text.replace("search", "").replace("wiki", "").strip()
        if query:
            response = await search_wikipedia(query)
        else:
            response = "What’s up?"
    
    elif "joke" in text:
        jokes = [
            "Robot’s day off? Circuit party!",
            "Coder snack? Bug-free chips!"
        ]
        response = random.choice(jokes)
    
    elif "brainstorm" in text:
        topic = text.replace("brainstorm", "").strip()
        if topic:
            response = await brainstorm_idea(topic)
        else:
            response = "Gimme a spark!"
    
    elif "goal" in text:
        goal = text.replace("goal", "").strip()
        if goal:
            response = await track_goal(goal)
        else:
            response = "Dream big, what’s it?"
    
    elif "create" in text or "make" in text:
        prompt_text = text.replace("create", "").replace("make", "").strip()
        if prompt_text:
            response = await spark_creativity(prompt_text)
        else:
            response = "What’s cooking?"
    
    else:
        response = await generate_response(text, conversation_context)
        conversation_context.append({"role": "assistant", "content": response})
    
    return response, conversation_context

async def main():
    conversation_context = []
    greeting = "Yo, Sam here! Let’s chat or build epic stuff."
    print(greeting)
    text_to_speech(greeting)
    
    while True:
        try:
            audio_data = await record_audio()
            if audio_data is None:
                continue
                
            transcription = await transcribe_audio(audio_data)
            if not transcription or transcription.startswith(("Transcription", "Audio", "No speech")):
                print(f"Transcription issue: {transcription}")
                continue
                
            print(f"You said: {transcription}")
            
            response, conversation_context = await process_command(transcription, conversation_context)
            print(f"Sam says: {response}")
            
            text_to_speech(response)
            
        except Exception as e:
            error_msg = f"Oops, broke something: {str(e)}"
            print(error_msg)
            text_to_speech(error_msg)
        
        await asyncio.sleep(0.1)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == '__main__':
        asyncio.run(main())