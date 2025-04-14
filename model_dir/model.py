import speech_recognition as sr
import openai
from elevenlabs import generate, save
import os
import tempfile
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not openai.api_key or not ELEVENLABS_API_KEY:
    logger.error("API keys are missing. Please set OPENAI_API_KEY and ELEVENLABS_API_KEY in .env")
    raise EnvironmentError("Missing API keys")

def capture_audio():
    recognizer = sr.Recognizer()
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "input_audio.wav")

    try:
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source)
            logger.info("Listening for audio...")
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
        
        with open(temp_path, "wb") as f:
            f.write(audio.get_wav_data())
        logger.info("Audio saved to %s", temp_path)
        return temp_path
    except sr.WaitTimeoutError:
        logger.error("No audio detected within timeout")
        raise Exception("No audio detected")
    except Exception as e:
        logger.error("Microphone error: %s", str(e))
        raise Exception(f"Microphone error: {str(e)}")

def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as f:
            logger.info("Transcribing audio with Whisper")
            transcription = openai.Audio.transcribe(model="whisper-1", file=f)
        text = transcription['text']
        if not text:
            logger.error("Transcription failed")
            raise Exception("Transcription failed")
        logger.info("Transcription: %s", text)
        return text
    except Exception as e:
        logger.error("Transcription error: %s", str(e))
        raise Exception(f"Transcription error: {str(e)}")

def process_text(text):
    try:
        logger.info("Processing text with GPT")
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Provide a concise response."},
                {"role": "user", "content": text}
            ]
        )
        gpt_text = gpt_response.choices[0].message.content
        if not gpt_text:
            logger.error("GPT processing failed")
            raise Exception("GPT processing failed")
        logger.info("GPT response: %s", gpt_text)
        return gpt_text
    except Exception as e:
        logger.error("GPT error: %s", str(e))
        raise Exception(f"GPT error: {str(e)}")

def text_to_speech(text):
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "output.mp3")
    
    try:
        logger.info("Generating speech with ElevenLabs")
        audio_data = generate(
            text=text,
            voice="Rachel",
            model="eleven_monolingual_v1",
            api_key=ELEVENLABS_API_KEY
        )
        save(audio_data, output_path)
        logger.info("Audio saved to %s", output_path)
        return output_path
    except Exception as e:
        logger.error("TTS error: %s", str(e))
        raise Exception(f"TTS error: {str(e)}")

def main():
    try:
        audio_path = capture_audio()
        
        text = transcribe_audio(audio_path)
    
        response_text = process_text(text)
        output_audio = text_to_speech(response_text)
        if os.path.exists(audio_path):
            os.remove(audio_path)
            logger.info("Cleaned up input audio file")
        
        return output_audio
    except Exception as e:
        logger.error("Pipeline error: %s", str(e))
        raise

if __name__ == "__main__":
    try:
        output = main()
        print(f"Output audio saved at: {output}")
    except Exception as e:
        print(f"Error: {str(e)}")