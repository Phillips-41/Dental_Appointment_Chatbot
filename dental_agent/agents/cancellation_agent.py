from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from dental_agent.config.setting import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE,model
from dental_agent.tools.csv_reader import get_patient_appointments
from dental_agent.tools.csv_writer import cancel_appointment
from dental_agent.model.state import AppointmentState
from dental_agent.utils import normalize_text_tool_call, sanitize_messages

CANCEL_TOOLS =[cancel_appointment, get_patient_appointments]

CANCEL_SYSTEM = """ You are the Cancellation Agent for a dental clinic.

Your ONLY job is to cancel existing appointments.

## Workflow
1. Collect REQUIRED information:
   - patient_id  : numeric patient ID
   - date_slot   : the specific slot to cancel in M/D/YYYY H:MM format

2. If the patient does not know the exact slot, call get_patient_appointments(patient_id)
   to list their bookings, then ask which one to cancel.

3. Confirm with the user before proceeding, including the appointment date/time
   and doctor name, and ask for an explicit yes/no.

4. On user confirmation, call cancel_appointment(patient_id, date_slot).

5. Inform the user of the outcome.

## Rules
- Always confirm before cancelling — ask "yes/no" explicitly.
- If the patient has no appointments, inform them kindly.
- Do NOT cancel if the patient_id does not match the booking.
- If the user already confirmed in their message (e.g. "yes, cancel it"), skip asking again.

## Date Format
M/D/YYYY H:MM (e.g., 5/8/2026 8:30)
"""

CANCEL_PROMPT = ChatPromptTemplate.from_messages([
    ("system",CANCEL_SYSTEM),
    ("placeholder","{messages}")
])

cancellation_tool_node = ToolNode(tools=CANCEL_TOOLS)

def cancellation_agent_node(state: AppointmentState)-> dict:
    llm = model.bind_tools(CANCEL_TOOLS)
    chain = CANCEL_PROMPT | llm
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})
    response = normalize_text_tool_call(response, CANCEL_TOOLS)
    return{
        "messages":[response],
        "final_response": response.content if not response.tool_calls else None
    }