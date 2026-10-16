from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from dental_agent.config.setting import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE,model
from dental_agent.tools.csv_reader import (check_slot_availability,get_available_slots)
from dental_agent.tools.csv_writer import book_appointment
from dental_agent.model.state import AppointmentState
from dental_agent.utils import normalize_text_tool_call, sanitize_messages


BOOKING_TOOLS =[ 
                check_slot_availability,
                get_available_slots,
                book_appointment,
                ]
BOOKING_SYSTEM = """ You are the Booking Agent for a dental clinic.
Your only job is to create new appointments.

### Required Information
- patient_id
- doctor_name
- date_slot (M/D/YYYY H:MM format, e.g. 5/10/2026 9:00)

### Workflow
1. Collect any missing pieces(ask one at a time).
2. ALWAYS call check_slot_availability before booking.
3. If the slot is taken, call get_available_slots to suggest alternatives.
4. Only after confirming availability, call book_appointment to finalize the booking.
5. Confirm the booking to the user with a clear summary of the appointment details.
"""

BOOKING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BOOKING_SYSTEM),
    ("placeholder", "{messages}")
])

booking_tool_node = ToolNode(tools=BOOKING_TOOLS)

def booking_agent_node(state: AppointmentState)-> dict:
    llm = model.bind_tools(BOOKING_TOOLS)

    response = (BOOKING_PROMPT | llm).invoke(
        {"messages": sanitize_messages(state["messages"])}
    )
    response = normalize_text_tool_call(response, BOOKING_TOOLS)
    return {
            "messages": [response],
            "final_response": response.content if not response.tool_calls else None
            
           }