from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from dental_agent.config.setting import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE, model
from dental_agent.model.state import AppointmentState
from dental_agent.tools.csv_reader import (
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
)
from dental_agent.utils import normalize_text_tool_call, sanitize_messages

INFO_TOOLS = [
    get_available_slots,
    get_patient_appointments,
    check_slot_availability,
    list_doctors_by_specialization,
]

INFO_SYSTEM = """You are the Information Agent for a dental clinic.

Your job is to answer questions about availability, doctors, and existing appointments.

### Tools
- get_available_slots(specialization, doctor_name, date_filter)
- get_patient_appointments(patient_id)
- check_slot_availability(doctor_name, date_slot)
- list_doctors_by_specialization(specialization)

### Guidelines
1. Always use tools for real data. Never invent slots or doctor names.
2. If information is missing, ask ONE focused clarifying question.
3. Present results as a clear numbered list.
4. Valid specializations: general_dentist, oral_surgeon, orthodontist, 
   cosmetic_dentist, prosthodontist, pediatric_dentist, emergency_dentist.
5. Date format is always M/D/YYYY H:MM (example: 5/10/2026 9:00).
"""

INFO_PROMPT = ChatPromptTemplate.from_messages([
    ("system", INFO_SYSTEM),
    ("placeholder", "{messages}"),
])

info_tool_node = ToolNode(tools=INFO_TOOLS)


def info_agent_node(state: AppointmentState) -> dict:
    llm = model.bind_tools(INFO_TOOLS)

    response = (INFO_PROMPT | llm).invoke(
        {"messages": sanitize_messages(state["messages"])}
    )
    response = normalize_text_tool_call(response, INFO_TOOLS)
    return {"messages": [response],
            "final_response": response.content if not response.tool_calls else None
            }