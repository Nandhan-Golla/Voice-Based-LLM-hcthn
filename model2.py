
import os
import asyncio
from groq import AsyncGroq
from gtts import gTTS
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

#groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
groq_client = AsyncGroq(api_key="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")
pygame.mixer.init()

SAMPLE_RATE = 16000
FRAME_DURATION = 0.03
SPEECH_THRESHOLD = 0.6

async def record_audio():
    print("Listening...")
    recording = []
    silence_frames = 0
    max_silence_frames = int(1.0 / FRAME_DURATION) 
    
    def callback(indata, frames, time, status):
        nonlocal recording, silence_frames
        if status:
            print(status)
        recording.append(indata.copy())
        energy = np.sum(indata ** 2) / len(indata)
        if energy < SPEECH_THRESHOLD:
            silence_frames += 1
        else:
            silence_frames = 0
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=int(SAMPLE_RATE * FRAME_DURATION)):
        while silence_frames < max_silence_frames:
            await asyncio.sleep(FRAME_DURATION)
    
    if recording:
        return np.concatenate(recording, axis=0)
    return None

async def transcribe_audio(audio_data):
    """Transcribe audio using Groq's Whisper model."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
        wavfile.write(temp_wav.name, SAMPLE_RATE, audio_data)
        
        with open(temp_wav.name, 'rb') as audio_file:
            response = await groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file,
                response_format="text"
            )
        
        os.unlink(temp_wav.name)
        return response.strip()

async def generate_response(text):
    """Generate a response using Groq's LLM."""
    try:
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are JARVIS, a helpful AI assistant inspired by Iron Man. Provide clear, concise, and accurate responses with a touch of wit."},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Sorry, sir, I'm having a bit of a malfunction: {str(e)}"

def text_to_speech(text):
    """Convert text to speech using gTTS."""
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_mp3:
            tts.save(temp_mp3.name)
            pygame.mixer.music.load(temp_mp3.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            os.unlink(temp_mp3.name)
    except Exception as e:
        print(f"Speech synthesis error: {str(e)}")

async def get_weather(city):
    """Fetch weather information for a given city."""
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Weather service unavailable. Please configure an OpenWeather API key."
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"Sorry, I couldn't fetch the weather for {city}."
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"The weather in {city} is {description} with a temperature of {temp}°C."
    except Exception as e:
        return f"Error fetching weather: {str(e)}"

async def search_wikipedia(query):
    """Search Wikipedia for a summary."""
    try:
        summary = wikipedia.summary(query, sentences=2)
        return f"Here's what I found on Wikipedia: {summary}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Your query is ambiguous. Could you be more specific? Options include: {', '.join(e.options[:3])}"
    except Exception as e:
        return f"Sorry, I couldn't find that on Wikipedia: {str(e)}"

async def process_command(text):
    """Process the user's command and return an appropriate response."""
    text = text.lower().strip()
    
    if not text:
        return "I didn't catch that, sir. Could you repeat?"
    
    if "time" in text:
        current_time = datetime.now().strftime("%H:%M:%S")
        return f"The current time is {current_time}."
    
    if "weather" in text:
        city = text.replace("weather", "").replace("in", "").strip() or "London"
        return await get_weather(city)
    
    if "search" in text or "wiki" in text:
        query = text.replace("search", "").replace("wiki", "").strip()
        if query:
            return await search_wikipedia(query)
        return "Please specify what to search for."
    
    if "joke" in text:
        return "Why did the computer go to art school? Because it wanted to learn how to draw a better 'byte'!"
    
    return await generate_response(text)

async def main():
    """Main loop for the voice assistant."""
    print("JARVIS online. How may I assist you, sir?")
    text_to_speech("JARVIS online. How may I assist you, sir?")
    
    while True:
        try:
            audio_data = await record_audio()
            if audio_data is None or len(audio_data) == 0:
                continue
                
            transcription = await transcribe_audio(audio_data)
            if not transcription:
                continue
                
            print(f"You said: {transcription}")
            
            response = await process_command(transcription)
            print(f"JARVIS: {response}")
    
            text_to_speech(response)
            
        except Exception as e:
            error_msg = f"Apologies, sir, I encountered an error: {str(e)}"
            print(error_msg)
            text_to_speech(error_msg)
        
        await asyncio.sleep(0.1)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
