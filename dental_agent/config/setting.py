import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = str(BASE_DIR / "doctor_availability.csv")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

MODEL_NAME = "gemini-3.8-flash"

MODEL ="qwen2.5-coder:7b"
TEMPERATURE = 0

VALID_SPECIALIZATIONS = [
    "general_dentist", "oral_surgeon", "orthodontist",
    "cosmetic_dentist", "prosthodontist", "pediatric_dentist",
    "emergency_dentist",
]

DATE_FORMAT = "%m/%d/%Y %H:%M"


model = ChatOllama(model=MODEL)

model1=ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=2048
    )