import os
import re
import json
import time
import wave
import base64
import asyncio
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional

import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel # Pydantic BaseModel is used implicitly by FastAPI, but not explicitly defined here
from groq import Groq
import edge_tts
import chromadb
from chromadb.utils import embedding_functions
from deepface import DeepFace 

# =====================================================
# 1. CONFIGURATION & MULTILINGUAL SETUP
# =====================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | [%(name)s] | %(message)s',
    handlers=[
        logging.FileHandler("neo_core_enterprise.log", encoding='utf-8'), 
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Neo-OS")

class Config:
    # --- 🔑 API KEYS ---
    # Load from environment variables for security in production
    GROQ_KEY = os.getenv("GROQ_KEY") 
    
    # --- 🧠 AI MODELS ---
    # Updated to recent Groq models, ensure these are available on Groq's platform
    VISION_MODEL = "llama-3.1-405b-reasoning" 
    LLM_MODEL = "llama-3.1-70b-versatile"                     
    STT_MODEL = "whisper-large-v3" # Groq's STT model often does not have 'turbo' suffix

    # --- 🌍 MULTILINGUAL VOICE MAPPING ---
    # These are Edge TTS voices
    DEFAULT_VOICE = "en-IN-NeerjaNeural"
    VOICE_MAP = {
        "hi": "hi-IN-SwaraNeural",        # Hindi
        "en": "en-IN-NeerjaNeural",       # English (Indian Accent)
        "mr": "mr-IN-AarohiNeural",       # Marathi
        "ta": "ta-IN-PallaviNeural",      # Tamil
        "te": "te-IN-ShrutiNeural",       # Telugu
        "gu": "gu-IN-DhwaniNeural",       # Gujarati
        "bn": "bn-IN-TanishaaNeural",     # Bengali
        "kn": "kn-IN-SapnaNeural",        # Kannada
        "ml": "ml-IN-SobhanaNeural",      # Malayalam
        "es": "es-ES-ElviraNeural",       # Spanish
        "fr": "fr-FR-DeniseNeural",       # French
        "de": "de-DE-AmalaNeural"         # German
    }

    # --- 📂 SYSTEM PATHS ---
    DATA_DIR = "system_data"
    LATEST_IMAGE = os.path.join(DATA_DIR, "current_view.jpg")
    TEMP_IMAGE = os.path.join(DATA_DIR, "temp_view.jpg") 

    # --- ⚙️ BEHAVIOR & ROBUSTNESS SETTINGS ---
    CHRONICLER_INTERVAL = 12   
    PROACTIVE_COOLDOWN = 60    
    FACE_THRESHOLD = 0.80      
    MAX_RETRIES = 3            
    MIN_AUDIO_BYTES = 1024     

# Directory Init
for d in ["voice_inputs", "ai_responses"]:
    os.makedirs(os.path.join(Config.DATA_DIR, d), exist_ok=True)

try:
    if not Config.GROQ_KEY:
        raise ValueError("GROQ_KEY environment variable not set. Please provide it for Groq client initialization.")
    groq_client = Groq(api_key=Config.GROQ_KEY, max_retries=2)
    logger.info("Groq Client initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize Groq Client: {e}")
    groq_client = None # Set to None to handle gracefully if key is missing or invalid

# Vector DB Init
logger.info("🧠 Initializing Neo's Neural Memory Databases...")
try:
    chroma_client = chromadb.PersistentClient(path="./neospecs_db")
    # Using 'all-MiniLM-L6-v2' as a default SentenceTransformer model, which needs sentence-transformers package.
    # On Render free tier, data in 'neospecs_db' will be ephemeral (deleted on restart).
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    visual_db = chroma_client.get_or_create_collection("visual_memory", embedding_function=emb_fn)
    face_db = chroma_client.get_or_create_collection("social_memory", embedding_function=emb_fn)
    logger.info("Neural Memory Databases initialized.")
except Exception as e:
    logger.error(f"⚠️ Vector DB Init Error (Will run degraded, memory will not persist): {e}")
    visual_db = None
    face_db = None

# =====================================================
# 2. REAL-TIME MULTILINGUAL APP STATE
# =====================================================

class AppState:
    SYSTEM_MODE: str = "professional"  
    IS_BUSY: bool = False
    IS_FIRST_INTERACTION: bool = True
    INTERACTION_HISTORY: List[Dict[str, str]] =[]
    
    CURRENT_LANGUAGE_CODE: str = "en"
    CURRENT_LANGUAGE_NAME: str = "English"
    
    LAST_PROACTIVE_TIME: float = 0.0
    PROACTIVE_AUDIO_PENDING: bool = False
    PROACTIVE_AUDIO_URL: str = ""
    LAST_PROCESSED_IMAGE_TIME: float = 0.0
    
    @classmethod
    def log_interaction(cls, user_text: str, ai_text: str):
        cls.INTERACTION_HISTORY.append({
            "user": user_text, "ai": ai_text, "time": datetime.now().strftime("%I:%M %p"), "mode": cls.SYSTEM_MODE # Include mode for Flutter logs
        })
        # Keep history concise to prevent token overflow for LLM context
        if len(cls.INTERACTION_HISTORY) > 6:
            cls.INTERACTION_HISTORY.pop(0)
            
    @classmethod
    def get_context_string(cls) -> str:
        if not cls.INTERACTION_HISTORY: return "No prior context."
        # Limit context to prevent token overflow with LLM/vision models
        return "\n".join([f"User: {x['user']} | Neo: {x['ai']}" for x in cls.INTERACTION_HISTORY[-3:]])

# =====================================================
# 3. ROBUST AUDIO & PLAYBACK CONTROLLER (REMOVED: Server does not play audio)
# =====================================================
# All pygame imports and AudioController class removed as server no longer plays audio.

# =====================================================
# 4. PREMIUM AI GENERATOR & PHONETIC INTERPRETER
# =====================================================

class AIService:
    @staticmethod
    def get_neo_prompt(mode: str, lang_name: str) -> str:
        base = (
            "You are Neo, an incredibly advanced, elegant, and professional AI assistant operating inside smart glasses (similar to Jarvis).\n"
            f"CRITICAL RULE 1: You MUST respond entirely in '{lang_name}'. Adapt perfectly to the language and cultural nuance.\n"
            "CRITICAL RULE 2: Keep it to 1-3 short, confident sentences. Be highly structured and factual.\n"
            "CRITICAL RULE 3: NEVER invent stories, make random assumptions, or hallucinate. If you don't know, say so professionally.\n"
            "CRITICAL RULE 4: No jokes unless explicitly requested. Maintain a premium, executive tone.\n"
        )
        if mode == "wingman":
            base += "MODE: Wingman. Analyze social cues and emotions factually. Provide brief, actionable insights without being informal."
        elif mode == "blind_assistant":
            base += "MODE: Blind Assistant. Prioritize absolute safety. Warmly and clearly guide the user away from obstacles. Be direct and concise."
        
        return f"{base}\nTime: {datetime.now().strftime('%I:%M %p')}\nRecent Context:\n{AppState.get_context_string()}"

    @staticmethod
    def clean_text(text: str) -> str:
        if not text: return "..."
        # Remove markdown, extra spaces, and common unwanted chars
        cleaned = re.sub(r'[*_`\[\]()<>]', '', text).strip()
        cleaned = re.sub(r'\s+', ' ', cleaned) # Replace multiple spaces with single
        return cleaned

    @staticmethod
    async def transcribe_with_retry(wav_path: str) -> str:
        if not groq_client:
            raise RuntimeError("Groq client not initialized. Cannot perform Speech-to-Text.")
        for attempt in range(Config.MAX_RETRIES):
            try:
                with open(wav_path, "rb") as audio_file:
                    transcript = await asyncio.to_thread(
                        lambda: groq_client.audio.transcriptions.create(
                            file=("audio.wav", audio_file.read()), 
                            model=Config.STT_MODEL,
                            prompt="Detect language accurately. Ensure proper formatting for Indian languages." 
                        )
                    )
                return transcript.text.strip()
            except Exception as e:
                logger.warning(f"STT Attempt {attempt+1} Failed: {e}")
                await asyncio.sleep(0.5) # Small delay before retrying
        raise RuntimeError("Speech-to-Text pipeline failed after multiple retries.")

    @staticmethod
    async def classify_and_correct_intent(user_query: str) -> Dict[str, Any]:
        if not groq_client:
            # Fallback if Groq client is not initialized
            return {"intent": "GENERAL_CHAT", "language_code": "en", "language_name": "English", "is_unclear": False, "corrected_query": user_query}
        
        """
        Fixes common transcription errors and classifies user intent.
        """
        prompt = (
            f"Analyze this transcribed speech from the user: '{user_query}'\n"
            "Speech recognition often makes mistakes with Indian locations or mixed languages (e.g., 'junior Tina and go distance' instead of 'Junnar to Pune distance').\n\n"
            "TASK:\n"
            "1. Try to deduce the actual meaning based on phonetic similarity and logical context.\n"
            "2. If the text is complete gibberish and cannot be deduced reliably, set 'is_unclear' to true.\n"
            "3. Determine the primary intended language (e.g., Marathi, Hindi, English, Hinglish). Output ISO 639-1 code and full name.\n"
            "4. Categorize intent into EXACTLY ONE of:[VISION, NAVIGATION, OBJECT_DESCRIPTION, GENERAL_CHAT, TRANSLATION, QUESTION, SAFETY_ALERT, UNCLEAR_SPEECH, LEARN_FACE].\n\n"
            "Output valid JSON ONLY:\n"
            "{\n"
            "  'intent': 'string',\n"
            "  'language_code': 'string (ISO 639-1 code)',\n"
            "  'language_name': 'string',\n"
            "  'is_unclear': boolean,\n"
            "  'corrected_query': 'string (the phonetically corrected logical query)'\n"
            "}"
        )
        try:
            res = await asyncio.to_thread(
                lambda: groq_client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.1, max_tokens=150
                )
            )
            data = json.loads(res.choices[0].message.content)
            
            # Failsafe overrides
            if data.get("is_unclear") is True:
                data["intent"] = "UNCLEAR_SPEECH"
            
            return {
                "intent": data.get("intent", "GENERAL_CHAT").upper(),
                "language_code": data.get("language_code", "en").lower(),
                "language_name": data.get("language_name", "English"),
                "is_unclear": data.get("is_unclear", False),
                "corrected_query": data.get("corrected_query", user_query)
            }
        except Exception as e:
            logger.error(f"Classification JSON Error: {e}")
            logger.debug(traceback.format_exc())
            return {"intent": "GENERAL_CHAT", "language_code": "en", "language_name": "English", "is_unclear": False, "corrected_query": user_query}

    @staticmethod
    async def generate_dynamic_response(context_prompt: str) -> str:
        if not groq_client:
            return "AI services are currently offline. Please check the API key."
        sys_prompt = AIService.get_neo_prompt(AppState.SYSTEM_MODE, AppState.CURRENT_LANGUAGE_NAME)
        try:
            res = await asyncio.to_thread(
                lambda: groq_client.chat.completions.create(
                    model=Config.LLM_MODEL,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": context_prompt}
                    ],
                    temperature=0.5, max_tokens=150
                )
            )
            return AIService.clean_text(res.choices[0].message.content)
        except Exception as e:
            logger.error(f"LLM Fault: {e}")
            logger.debug(traceback.format_exc())
            return "Experiencing a brief neural delay. One moment."

    @staticmethod
    async def generate_voice(text: str, output_path: str, lang_code: str = "en") -> bool:
        if not text or len(text.strip()) < 2:
            logger.warning("Attempted to generate voice for empty or too short text, skipping.")
            return False

        voice = Config.VOICE_MAP.get(lang_code[:2], Config.DEFAULT_VOICE)
        try:
            communicate = edge_tts.Communicate(text, voice, rate="+5%", pitch="-2Hz")
            await communicate.save(output_path)
            return True
        except Exception as e:
            logger.error(f"TTS Engine Failure: {e}")
            logger.debug(traceback.format_exc())
            return False

    @staticmethod
    def get_base64_image() -> Optional[str]:
        if not os.path.exists(Config.LATEST_IMAGE): 
            logger.debug(f"Image file not found: {Config.LATEST_IMAGE}")
            return None
        try:
            size = os.path.getsize(Config.LATEST_IMAGE)
            if size < 2048: # Minimum size to consider it a valid image
                logger.warning(f"Image too small ({size} bytes) in {Config.LATEST_IMAGE}, skipping base64 encoding.")
                return None
            with open(Config.LATEST_IMAGE, "rb") as img: 
                return base64.b64encode(img.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error reading or encoding image from {Config.LATEST_IMAGE}: {e}")
            logger.debug(traceback.format_exc())
            return None

    @staticmethod
    def safe_write_audio(audio_data: bytes, path: str) -> bool:
        if len(audio_data) < Config.MIN_AUDIO_BYTES: 
            logger.warning(f"Received audio data too small: {len(audio_data)} bytes. Minimum is {Config.MIN_AUDIO_BYTES}.")
            return False 
        try:
            # Assuming raw audio data is sent as application/octet-stream
            # If the client sends WAV headers, it might start with b'RIFF'
            # If not, we assume raw PCM and add headers.
            if audio_data.startswith(b'RIFF'):
                with open(path, "wb") as f: f.write(audio_data)
            else:
                # Assuming 16-bit, 1 channel, 16000 Hz PCM for raw data
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 2 bytes for 16-bit audio
                    wf.setframerate(16000) # Standard sample rate
                    wf.writeframes(audio_data)
            return True
        except Exception as e:
            logger.error(f"Audio write/validation failed for path {path}: {e}")
            logger.debug(traceback.format_exc())
            return False

# =====================================================
# 5. PROACTIVE AI ENGINE (Safety & Obstacles Focused)
# =====================================================

async def _chronicler_task():
    if not os.path.exists(Config.LATEST_IMAGE): 
        logger.debug("No latest image found for chronicler task.")
        return
    
    current_img_time = os.path.getmtime(Config.LATEST_IMAGE)
    if current_img_time <= AppState.LAST_PROCESSED_IMAGE_TIME: 
        logger.debug("Image not updated since last chronicler task check, skipping.")
        return 
    
    AppState.LAST_PROCESSED_IMAGE_TIME = current_img_time
    b64 = AIService.get_base64_image()
    if not b64: 
        logger.warning("Could not get base64 image for chronicler task (too small/error), skipping.")
        return

    try:
        if not groq_client:
            logger.warning("Chronicler task skipped: Groq client not initialized.")
            return

        prompt = (
            f"You are Neo. Time is {datetime.now().strftime('%I:%M %p')}. "
            "Scan the image STRICTLY for: immediate physical danger, incoming vehicles, stairs, low laptop battery, or someone approaching the user directly.\n"
            f"If an alert is needed, write EXACTLY 1 short, professional warning sentence IN {AppState.CURRENT_LANGUAGE_NAME}.\n"
            "If the scene is completely normal and safe, output EXACTLY the word 'SILENT'."
        )
        
        res = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                model=Config.VISION_MODEL,
                messages=[{"role": "user", "content":[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
                max_tokens=40, temperature=0.3
            )
        )
        ai_thought = res.choices[0].message.content.strip()
        current_time = time.time()
        
        if "SILENT" in ai_thought.upper() and len(ai_thought) < 10: # Only "SILENT" or very short "SILENT."
            logger.debug("Chronicler detected normal/safe scene.")
            # Do not add simple "SILENT" to memory
        else: # If it's an actual alert or more descriptive than just "SILENT"
            if visual_db: # Only add to memory if DB is initialized
                visual_db.add(documents=[f"[{datetime.now().strftime('%H:%M')}] {ai_thought}"], ids=[f"v_{current_time}"])
            
            if "SILENT" not in ai_thought.upper(): # This is a real alert
                if (current_time - AppState.LAST_PROACTIVE_TIME) > Config.PROACTIVE_COOLDOWN:
                    logger.info(f"👁️ Neo Proactive Alert[{AppState.CURRENT_LANGUAGE_NAME}]: {ai_thought}")
                    AppState.LAST_PROACTIVE_TIME = current_time
                    
                    filename = f"pro_{int(current_time)}.mp3"
                    path = os.path.join(Config.DATA_DIR, "ai_responses", filename)
                    
                    if await AIService.generate_voice(ai_thought, path, AppState.CURRENT_LANGUAGE_CODE):
                        AppState.PROACTIVE_AUDIO_PENDING = True
                        AppState.PROACTIVE_AUDIO_URL = f"/media/{filename}"
                        # REMOVED SERVER-SIDE PLAYBACK: threading.Thread(target=AudioController.play, args=(path,), daemon=True).start()
                        logger.info(f"Proactive audio generated for client: {AppState.PROACTIVE_AUDIO_URL}")
                    else:
                        logger.warning("Failed to generate proactive voice, no audio URL will be available for client.")

    except Exception as e: 
        logger.warning(f"Chronicler Background Error: {e}")
        logger.debug(traceback.format_exc()) # Log full traceback for debugging

async def background_loop():
    logger.info("📡 Neo Proactive Safety Sensors Online.")
    while True:
        await _chronicler_task()
        await asyncio.sleep(Config.CHRONICLER_INTERVAL)


# =====================================================
# 6. FASTAPI CORE & ENDPOINTS
# =====================================================

app = FastAPI(title="Neo Enterprise OS", version="8.0.0 (Professional Core)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/media", StaticFiles(directory=Config.DATA_DIR), name="media")

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(background_loop())
    try:
        if not groq_client:
            logger.warning("Skipping startup audio: Groq client not initialized.")
            return

        startup_path = os.path.join(Config.DATA_DIR, "ai_responses", "startup.mp3")
        txt = await AIService.generate_dynamic_response("System successfully booted. Introduce yourself professionally as Neo in 1 short sentence.")
        if await AIService.generate_voice(txt, startup_path, AppState.CURRENT_LANGUAGE_CODE):
            # REMOVED SERVER-SIDE PLAYBACK: threading.Thread(target=AudioController.play, args=(startup_path,), daemon=True).start()
            logger.info(f"Startup audio generated for client: /media/{os.path.basename(startup_path)}")
        else:
            logger.warning("Failed to generate startup voice.")
    except Exception as e:
        logger.error(f"Startup Audio Error: {e}")
        logger.debug(traceback.format_exc())


@app.post("/upload_image")
async def upload_image(request: Request):
    try:
        content = await request.body()
        if len(content) < 1024: 
            logger.warning(f"Received image too small: {len(content)} bytes. Minimum is 1024 bytes.")
            return JSONResponse({"status": "error", "message": "Image too small"}, status_code=400)
        
        with open(Config.TEMP_IMAGE, "wb") as f: f.write(content)
        os.replace(Config.TEMP_IMAGE, Config.LATEST_IMAGE)
        logger.info(f"Image uploaded and saved to {Config.LATEST_IMAGE}")
        return {"status": "ok"}
    except Exception as e: 
        logger.error(f"Image Upload Failed: {e}")
        logger.debug(traceback.format_exc())
        return JSONResponse({"status": "error", "message": "Failed to process image upload"}, status_code=500)

@app.get("/app/status")
async def get_app_status():
    return JSONResponse(content={
        "status": "success",
        "data": {
            "battery": random.randint(85, 100), 
            "status": "Processing..." if AppState.IS_BUSY else "Online & Ready",
            "mode": AppState.SYSTEM_MODE,
            "language": AppState.CURRENT_LANGUAGE_NAME,
            "is_busy": AppState.IS_BUSY,
            "uptime": datetime.now().strftime("%I:%M %p")
        }
    })

@app.get("/app/history")
async def get_app_history():
    return JSONResponse(content={"status": "success", "data": AppState.INTERACTION_HISTORY})

@app.get("/media/current_view.jpg")
async def get_live_camera():
    if not os.path.exists(Config.LATEST_IMAGE):
        return JSONResponse({"status": "error", "message": "Camera offline"}, status_code=404)
    headers = {"Cache-Control": "no-cache, no-store, must-revalidate", "Pragma": "no-cache"}
    return FileResponse(Config.LATEST_IMAGE, media_type="image/jpeg", headers=headers)

@app.get("/check_proactive")
async def check_proactive():
    if AppState.PROACTIVE_AUDIO_PENDING:
        resp = {"status": "success", "new_audio": True, "url": AppState.PROACTIVE_AUDIO_URL}
        AppState.PROACTIVE_AUDIO_PENDING = False # Reset the flag after client checks
        logger.info(f"Client checked for proactive audio, returning: {AppState.PROACTIVE_AUDIO_URL}")
        return JSONResponse(content=resp)
    return JSONResponse(content={"status": "success", "new_audio": False})

@app.get("/set_mode")
async def set_mode(mode: str): # Removed background_tasks as server won't play audio
    mode = mode.lower()
    if mode not in ["professional", "wingman", "blind_assistant"]: 
        return JSONResponse({"status": "error", "message": "Invalid Mode"}, status_code=400)
    
    # AudioController.stop() Removed server-side stop, Flutter app will handle stopping playback if needed.
    AppState.SYSTEM_MODE = mode
    
    mode_powers = {
        "professional": "reading text, analytical vision, and professional assistance",
        "wingman": "detecting emotions and analyzing social cues",
        "blind_assistant": "real-time hazard detection and obstacle avoidance"
    }
    
    prompt = f"User activated '{mode.replace('_', ' ').title()}' mode. Acknowledge this professionally in exactly 1 sentence, noting capabilities: {mode_powers[mode]}."
    filename = f"mode_{int(time.time())}.mp3"
    path = os.path.join(Config.DATA_DIR, "ai_responses", filename)
    dynamic_msg = await AIService.generate_dynamic_response(prompt)
    
    audio_url = None
    if await AIService.generate_voice(dynamic_msg, path, AppState.CURRENT_LANGUAGE_CODE):
        audio_url = f"/media/{os.path.basename(path)}"
        logger.info(f"Mode change audio generated for client: {audio_url}")
        # REMOVED SERVER-SIDE PLAYBACK: background_tasks.add_task(AudioController.play, path)
    
    # Return audio_url in the response so Flutter app can play it
    return JSONResponse({"status": "success", "data": {"mode": AppState.SYSTEM_MODE, "announcement": dynamic_msg, "audio_url": audio_url}})


# =====================================================
# 7. 🧠 THE MASTER INTELLIGENCE ROUTER
# =====================================================

@app.post("/process_voice")
async def process_voice(request: Request): # Removed background_tasks as server won't play audio
    if AppState.IS_BUSY:
        return JSONResponse({"status": "busy", "message": "Neo is currently processing another request."}, status_code=429)
        
    AppState.IS_BUSY = True
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = os.path.join(Config.DATA_DIR, "voice_inputs", f"rec_{ts}.wav")
    audio_out_filename = f"res_{ts}.mp3"
    audio_out_path = os.path.join(Config.DATA_DIR, "ai_responses", audio_out_filename)
    
    ai_text_response = "My systems experienced an unexpected interruption. Attempting to recover."
    returned_audio_url = None

    try:
        # 1. Capture & Validate Audio
        audio_data = await request.body()
        if not AIService.safe_write_audio(audio_data, wav_path):
            return JSONResponse({"status": "empty", "message": "Audio too short or invalid."}, status_code=400)
        
        # 2. Raw Transcription
        raw_query = await AIService.transcribe_with_retry(wav_path)
        if len(raw_query) < 2: 
            return JSONResponse({"status": "empty", "message": "No speech detected."}, status_code=200) # 200 is fine for no speech
            
        logger.info(f"🎤 Raw STT Input: {raw_query}")
        
        if any(word in raw_query.lower() for word in["stop", "quiet", "chup", "ruko"]):
            # AudioController.stop() Removed server-side stop
            # Generate a brief stop confirmation audio, so the user knows it registered
            stop_msg = "Command registered. Halting current processes."
            if await AIService.generate_voice(stop_msg, audio_out_path, AppState.CURRENT_LANGUAGE_CODE):
                returned_audio_url = f"/media/{audio_out_filename}"
            return JSONResponse({"status": "stopped", "ai_text": stop_msg, "audio_url": returned_audio_url})

        # 3. AI Phonetic Interpretation & Intent Classification
        analysis = await AIService.classify_and_correct_intent(raw_query)
        intent = analysis["intent"]
        lang_code = analysis["language_code"]
        lang_name = analysis["language_name"]
        is_unclear = analysis["is_unclear"]
        corrected_query = analysis["corrected_query"]

        AppState.CURRENT_LANGUAGE_CODE = lang_code
        AppState.CURRENT_LANGUAGE_NAME = lang_name
        
        logger.info(f"🧠 Analysis -> Intent: {intent} | Lang: {lang_name} | Corrected: {corrected_query}")

        b64_image = AIService.get_base64_image()
        ai_reply = ""

        # --- 4. EXECUTE MASTER LOGIC ---
        try:
            if is_unclear or intent == "UNCLEAR_SPEECH":
                ai_reply = await AIService.generate_dynamic_response(
                    f"The audio transcription was very unclear (raw: {raw_query}). Politely ask the user to repeat or clarify what they meant."
                )

            elif intent in ["VISION", "OBJECT_DESCRIPTION"]:
                if b64_image:
                    if not groq_client:
                        ai_reply = "Vision services are unavailable. Groq client not initialized."
                    else:
                        prompt = f"Analyze the image factually and structurally. Answer the user's query: '{corrected_query}'. Keep it to 2-3 sentences. No assumptions."
                        res = await asyncio.to_thread(lambda: groq_client.chat.completions.create(model=Config.VISION_MODEL, messages=[{"role": "user", "content":[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}]}], max_tokens=150).choices[0].message.content)
                        ai_reply = await AIService.generate_dynamic_response(f"Vision output: {res}. Deliver this cleanly to the user.")
                else:
                    ai_reply = await AIService.generate_dynamic_response("Inform the user professionally that the camera feed is currently offline.")

            elif intent == "NAVIGATION":
                ai_reply = await AIService.generate_dynamic_response(f"User requested navigation or distance info: '{corrected_query}'. Provide a highly professional, factual response acknowledging the request. Provide approximate logical distances if known, otherwise state you are calculating.")

            elif intent == "TRANSLATION":
                if b64_image and ("read" in corrected_query.lower() or "look" in corrected_query.lower()):
                    if not groq_client:
                        ai_reply = "Translation services are unavailable. Groq client not initialized."
                    else:
                        prompt = f"Read the visible text in the image and translate it based on this request: '{corrected_query}'"
                        res = await asyncio.to_thread(lambda: groq_client.chat.completions.create(model=Config.VISION_MODEL, messages=[{"role": "user", "content":[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}]}], max_tokens=150).choices[0].message.content)
                        ai_reply = await AIService.generate_dynamic_response(f"Translation result: {res}. Deliver it to the user.")
                else:
                    ai_reply = await AIService.generate_dynamic_response(f"Translate or respond to this translation request professionally: '{corrected_query}'")

            elif intent == "LEARN_FACE":
                if b64_image:
                    if not groq_client:
                        ai_reply = "Face learning services are unavailable. Groq client not initialized."
                    elif not face_db:
                        ai_reply = "Social memory database is not active. Cannot save face."
                    else:
                        info_ext = await asyncio.to_thread(lambda: groq_client.chat.completions.create(model=Config.LLM_MODEL, messages=[{"role":"user", "content":f"Extract name/context from: '{corrected_query}'. Output EXACTLY: Name - Context"}], max_tokens=20).choices[0].message.content)
                        # DeepFace might require specific TensorFlow/OpenCV builds which consume significant resources.
                        # This operation is resource-intensive and might fail on free tiers.
                        face_res = await asyncio.to_thread(lambda: DeepFace.represent(img_path=Config.LATEST_IMAGE, model_name="Facenet", enforce_detection=True))
                        face_db.add(documents=[info_ext], embeddings=[face_res[0]["embedding"]], ids=[f"person_{ts}"])
                        ai_reply = await AIService.generate_dynamic_response(f"Face successfully saved as '{info_ext}'. Confirm this professionally.")
                else: 
                    ai_reply = await AIService.generate_dynamic_response("Inform the user the camera is offline, so you cannot save the face.")

            elif intent == "SAFETY_ALERT":
                ai_reply = await AIService.generate_dynamic_response(f"User is asking about safety or danger: '{corrected_query}'. Assess and respond strictly regarding safety protocols and immediate environment.")

            elif intent == "QUESTION":
                if not groq_client:
                    ai_reply = "Intelligence services are unavailable. Groq client not initialized."
                else:
                    mem_context = "No recent visual memory relevant to this."
                    if visual_db: # Only query if DB is initialized
                        results = visual_db.query(query_texts=[corrected_query], n_results=2)
                        mem = results['documents'][0] if (results['documents'] and len(results['documents'][0]) > 0) else []
                        mem_context = " | ".join(mem) if mem else "No recent visual memory relevant to this."
                    
                    ai_reply = await AIService.generate_dynamic_response(f"User asks: '{corrected_query}'. Relevant camera memory context: '{mem_context}'. Answer the user factually based on memory or general intelligence.")

            else: 
                # GENERAL_CHAT (Default intent)
                ai_reply = await AIService.generate_dynamic_response(f"User states: '{corrected_query}'. Respond naturally and professionally.")

        except Exception as routing_error:
            logger.warning(f"Feature Route Error ({intent}): {routing_error}")
            logger.debug(traceback.format_exc())
            ai_reply = await AIService.generate_dynamic_response("You encountered an internal logic error. Apologize professionally and suggest trying again.")
        
        ai_text_response = ai_reply # Update the response text

        # 5. Generate Output Audio for Client
        logger.info(f"✨ Neo [{lang_code}]: {ai_reply}")
        AppState.log_interaction(corrected_query, ai_reply)

        if await AIService.generate_voice(ai_reply, audio_out_path, AppState.CURRENT_LANGUAGE_CODE):
            returned_audio_url = f"/media/{audio_out_filename}"
            logger.info(f"Response audio generated for client: {returned_audio_url}")
        else:
            logger.warning("Failed to generate voice response for client.")
        
        return JSONResponse({
            "status": "success", 
            "data": {
                "user_text": corrected_query, 
                "ai_text": ai_reply, 
                "audio_url": returned_audio_url, # Send the URL even if it's None (failed to generate)
                "intent": intent,
                "language": lang_name,
                "mode": AppState.SYSTEM_MODE 
            }
        })

    except Exception as e:
        logger.error(f"❌ Critical Core Fault during voice processing: {e}")
        logger.error(traceback.format_exc())
        
        fallback_msg_text = "My systems experienced an unexpected interruption. Attempting to recover."
        try: 
            if groq_client:
                # Try to get a professional apology from LLM if client is available
                fallback_msg_text = await AIService.generate_dynamic_response("System threw a critical error. Apologize professionally and suggest trying again.")
        except: 
            pass # Even dynamic response might fail in severe cases
        
        # Try to generate fallback audio if possible
        if await AIService.generate_voice(fallback_msg_text, audio_out_path, AppState.CURRENT_LANGUAGE_CODE):
            returned_audio_url = f"/media/{audio_out_filename}"
            logger.info(f"Fallback audio generated for client: {returned_audio_url}")

        return JSONResponse({"status": "error", "message": "Glitch Recovered", "ai_text": fallback_msg_text, "audio_url": returned_audio_url}, status_code=500)
    
    finally:
        AppState.IS_BUSY = False

# This block is for local development, Render uses Procfile
# if __name__ == "__main__":
#     logger.info("Starting Professional Enterprise Neo Server (Local Development Mode)...")
#     uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False, workers=1)
