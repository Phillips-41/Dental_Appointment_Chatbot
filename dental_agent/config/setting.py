import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langfuse import observe
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = str(BASE_DIR / "doctor_availability.csv")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
OPENAI_MODEL = os.getenv("OPENAI_MODEL")


if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

MODEL_NAME = "gemini-3.5-flash"

MODEL ="qwen2.5-coder:7b"
TEMPERATURE = 0

VALID_SPECIALIZATIONS = [
    "general_dentist", "oral_surgeon", "orthodontist",
    "cosmetic_dentist", "prosthodontist", "pediatric_dentist",
    "emergency_dentist",
]

DATE_FORMAT = "%m/%d/%Y %H:%M"


model1 = ChatOllama(model=MODEL)

model2=ChatGoogleGenerativeAI(
            model=MODEL_NAME,
            google_api_key=GEMINI_API_KEY,
            temperature=TEMPERATURE,
            max_output_tokens=2048
    )

model = ChatOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=base_url,
    model=OPENAI_MODEL,
    temperature=0
)