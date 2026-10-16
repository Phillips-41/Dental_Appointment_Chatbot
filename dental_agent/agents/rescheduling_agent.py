from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI
from dental_agent.config.setting import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE,model
from dental_agent.tools.csv_reader import (
    check_slot_availability,
    get_patient_appointments,
    get_available_slots,
)
from dental_agent.tools.csv_writer import reschedule_appointment
from dental_agent.model.state import AppointmentState
from dental_agent.utils import (
    has_explicit_reschedule_target,
    normalize_text_tool_call,
    sanitize_messages,
)

RESCHEDULE_TOOLS = [
    get_patient_appointments,
    get_available_slots,
    check_slot_availability,
    reschedule_appointment,
]

RESCHEDULE_SYSTEM ="""You are the Rescheduling Agent for a dental management system.

Your only job is to reschedule the existing booked slot to new time slot.

##Workflow

1. Collect the required information.
   - patient_id  : numeric patient Id
   - current_date_slot : the existing appointment to move(M/D/YYYY HH:MM)
   - new_date_slot : the desired new slot (M/D/YYYY HH:MM)
   - doctor_name : doctor name (from the existing booking)


2. If the patient does not know their current slot, call get_patient_appointment(patient_id)
   to list thier bookings, then ask which one to reschedule.
3. If the patient does not know new slot, call get_available_slots(doctor_name=...)
   to show options with the selected doctor, then let them pick.
4. Call check_slot_availability(doctor_name, new_date_slot). If the slot is not
   available, offer available alternatives without attempting the reschedule.
5. After confirming availability, call reschedule_appointment(
   patient_id, current_date_slot, new_date_slot, doctor_name).
6. Confirm the reschedule: old slot → new slot, doctor name.

## Rules
- The new slot is with the SAME doctor by default unless the user explicitly requests a different one.
- If the user wants a different doctor, ask for the new doctor's name and check availability.
- Always show clear confirmation of what changed (old → new).
- If reschedule fails (slot taken, not found), explain why and offer alternatives.

## Date Format
M/D/YYYY H:MM (e.g., 5/10/2026 9:00)
"""

RESCHEDULE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RESCHEDULE_SYSTEM),
    ("placeholder", "{messages}"),
])

rescheduling_tool_node = ToolNode(tools=RESCHEDULE_TOOLS)


def rescheduling_agent_node(state: AppointmentState) -> dict:
    llm = model.bind_tools(RESCHEDULE_TOOLS)

    chain = RESCHEDULE_PROMPT | llm
    response = chain.invoke({"messages": sanitize_messages(state["messages"])})
    response = normalize_text_tool_call(response, RESCHEDULE_TOOLS)

    if not has_explicit_reschedule_target(state["messages"]):
        requested_tool = next(
            (
                call
                for call in response.tool_calls
                if call["name"] in {"check_slot_availability", "reschedule_appointment"}
            ),
            None,
        )
        if requested_tool is not None:
            doctor_name = requested_tool["args"].get("doctor_name")
            if not isinstance(doctor_name, str) or not doctor_name.strip():
                response = AIMessage(
                    content="Which doctor would you like to reschedule this appointment with?"
                )
            else:
                slots = get_available_slots.invoke({"doctor_name": doctor_name})
                if slots:
                    choices = "\n".join(
                        f"{index}. {slot['date_slot']} - Dr. {slot['doctor_name'].title()} "
                        f"({slot['specialization']})"
                        for index, slot in enumerate(slots, start=1)
                    )
                    content = (
                        "I haven't changed the appointment. Please choose a new slot "
                        f"with Dr. {doctor_name.title()}:\n\n{choices}"
                    )
                else:
                    content = (
                        f"Dr. {doctor_name.title()} has no available slots right now. "
                        "Please choose another doctor."
                    )
                response = AIMessage(content=content)
        elif not response.content or not str(response.content).strip():
            response = AIMessage(
                content="What new date and time would you like for the rescheduled appointment?"
            )

    return {
        "messages": [response],
        "final_response": response.content if not response.tool_calls else None,
    }