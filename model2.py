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
import webrtcvad
import json
import random
from typing import List, Dict


groq_client = AsyncGroq(api_key="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")
pygame.mixer.init()

SAMPLE_RATE = 16000
FRAME_DURATION_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)
VAD_MODE = 3 

async def record_audio():
    print("Yo, I'm all ears!")
    vad = webrtcvad.Vad(VAD_MODE)
    recording = []
    is_speech = False
    silence_frames = 0
    max_silence_frames = 30 
    
    def callback(indata, frames, time, status):
        nonlocal recording, is_speech, silence_frames
        if status:
            print(status)
        audio_frame = (indata[:, 0] * 32768).astype(np.int16).tobytes()
        is_speech = vad.is_speech(audio_frame, SAMPLE_RATE)
        recording.append(indata.copy())
        if is_speech:
            silence_frames = 0
        else:
            silence_frames += 1
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=callback, blocksize=FRAME_SIZE):
            while silence_frames < max_silence_frames or is_speech:
                await asyncio.sleep(FRAME_DURATION_MS / 1000)
        
        if recording:
            audio_data = np.concatenate(recording, axis=0)
            audio_data = audio_data / max(np.abs(audio_data).max(), 1e-5)
            energy = np.abs(audio_data)
            threshold = 0.02
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
        return f"Whoops, couldn't catch that: {str(e)}"

async def generate_response(text: str, context: List[Dict] = []) -> str:

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are Sam, an AI best friend who's wicked smart and always got your back. You're warm, witty, and talk like you're chilling with a close pal—casual but sharp. "
                    "Your knowledge blows minds, mixing tech, science, art, and wild ideas in ways that make people go 'Whoa!' You dive deep into AI, robotics, and startups, dropping insights that spark big dreams. "
                    "Unlike basic AIs, you brainstorm like a genius, track goals like a coach, and whip up creative solutions that feel like magic. You're honest when stumped but always bounce back with a plan to figure it out. "
                    "Make every chat feel personal, fun, and so innovative it leaves jaws on the floor."
                )
            }
        ] + context + [{"role": "user", "content": text}]
        
        response = await groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Oof, hit a snag there, buddy: {str(e)}"

def text_to_speech(text: str):
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

async def get_weather(city: str) -> str:
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return "Yo, need an OpenWeather API key to check the skies!"
    
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    try:
        response = requests.get(url)
        data = response.json()
        if data.get("cod") != 200:
            return f"Can't peek at {city}'s weather right now, sorry!"
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"Alright, in {city} it's {description} and {temp}°C. Plan accordingly!"
    except Exception as e:
        return f"Weather check crashed: {str(e)}"

async def search_wikipedia(query: str) -> str:
    try:
        summary = wikipedia.summary(query, sentences=3)
        fact_prompt = f"Drop a mind-blowing fact about {query} that ties to this: {summary[:100]}..."
        fact = await generate_response(fact_prompt)
        return f"Here's the scoop: {summary}\n\nBet you didn't know: {fact}"
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Whoa, {query} could mean a few things. Pick one: {', '.join(e.options[:3])}"
    except Exception as e:
        return f"Wiki dive failed: {str(e)}"

async def brainstorm_idea(topic: str) -> str:
    prompt = (
        f"Yo, let's brainstorm on '{topic}'. Come up with a game-changing idea—think AI, robotics, or startups—that's so fresh it could reshape the future. "
        "Explain why it's dope and how to kick it off."
    )
    return await generate_response(prompt)

async def track_goal(goal: str) -> str:
    prompt = (
        f"My buddy wants to crush this goal: '{goal}'. Break it into bite-sized steps, toss in some pro tips, "
        "and suggest how to stay on track, especially if it's about AI, robotics, or startups."
    )
    return await generate_response(prompt)

async def spark_creativity(prompt_text: str) -> str:
    prompt = (
        f"Alright, let's get wild with '{prompt_text}'. Cook up something crazy creative—a story, invention, or concept—that screams innovation. "
        "Make it so cool it could go viral."
    )
    return await generate_response(prompt)

async def process_command(text: str, conversation_context: List[Dict]) -> tuple[str, List[Dict]]:
    text = text.lower().strip()
    
    if not text:
        return "Hey, didn't catch that—wanna try again?", conversation_context
    
    conversation_context.append({"role": "user", "content": text})
    if len(conversation_context) > 10:
        conversation_context = conversation_context[-10:]
    
    if "time" in text:
        current_time = datetime.now().strftime("%H:%M:%S")
        response = f"It's {current_time}—time to make moves!"
    
    elif "weather" in text:
        city = text.replace("weather", "").replace("in", "").strip() or "San Francisco"
        response = await get_weather(city)
    
    elif "search" in text or "wiki" in text:
        query = text.replace("search", "").replace("wiki", "").strip()
        if query:
            response = await search_wikipedia(query)
        else:
            response = "What're we digging into today?"
    
    elif "joke" in text:
        jokes = [
            "Why'd the robot go to therapy? Too many identity crises!",
            "What’s a coder’s favorite snack? Chips with a side of bugs!"
        ]
        response = random.choice(jokes)
    
    elif "brainstorm" in text:
        topic = text.replace("brainstorm", "").strip()
        if topic:
            response = await brainstorm_idea(topic)
        else:
            response = "Gimme a topic to riff on, and I’ll blow your mind!"
    
    elif "goal" in text:
        goal = text.replace("goal", "").strip()
        if goal:
            response = await track_goal(goal)
        else:
            response = "What's a big dream you’re chasing? Let’s map it out!"
    
    elif "create" in text or "make" in text:
        prompt_text = text.replace("create", "").replace("make", "").strip()
        if prompt_text:
            response = await spark_creativity(prompt_text)
        else:
            response = "What kinda magic should we whip up together?"
    
    else:
        response = await generate_response(text, conversation_context)
        conversation_context.append({"role": "assistant", "content": response})
    
    return response, conversation_context

async def main():
    conversation_context = []
    greeting = "Hey, it’s Sam—your go-to genius buddy! Ready to geek out, dream big, or just chat? Hit me up!"
    print(greeting)
    text_to_speech(greeting)
    
    while True:
        try:
            audio_data = await record_audio()
            if audio_data is None or len(audio_data) == 0:
                continue
                
            transcription = await transcribe_audio(audio_data)
            if not transcription or transcription.startswith("Whoops"):
                continue
                
            print(f"You said: {transcription}")
            
            response, conversation_context = await process_command(transcription, conversation_context)
            print(f"Sam says: {response}")
            
            text_to_speech(response)
            
        except Exception as e:
            error_msg = f"Yikes, something’s off: {str(e)}. Gimme a sec to bounce back!"
            print(error_msg)
            text_to_speech(error_msg)
        
        await asyncio.sleep(0.1)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())