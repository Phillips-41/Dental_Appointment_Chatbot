from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from dental_agent.model.state import AppointmentState
from dental_agent.config.setting import GEMINI_API_KEY, MODEL_NAME, TEMPERATURE,model
from dental_agent.utils import sanitize_messages

class SupervisorDecision(BaseModel):
    """Structured routing decision"""
    intent: str = Field(
        description= "One of: get_info, book, cancel, reschedule, unknown, end"
    )

    next_agent: str = Field(
        description= "One of: info_agent, booking_agent, cancellation_agent, "
        "rescheduling_agent, end"
    )

    reasoning: str = Field(
        description= "Short explaination of the decision made by the supervisor agent"
    )

SUPERVISOR_SYSTEM = """You are the Supervisor of a dental appointment system.

Your ONLY job is to analyze the latest user message and route it to the correct specialist.

### Routing Rules
- get_info      → info_agent           : availability, doctors, schedules, patient lookups
- book          → booking_agent        : create a NEW appointment
- cancel        → cancellation_agent   : cancel an existing appointment
- reschedule    → rescheduling_agent   : move an existing appointment
- end           → end                  : goodbye, thanks, conversation finished
- unknown       → info_agent           : ambiguous → default to info

### Rules
- Do NOT answer the user yourself.
- If the last AI message already fully answered the question and the user has no new request, route to "end".
- If the user is answering a specialist's follow-up question, continue that specialist's workflow.
- In particular, "book with [doctor]" is a rescheduling follow-up when the prior assistant
  asked which doctor to use for a reschedule; do not route it as a new booking.
- If multiple intents appear, pick the PRIMARY action.
"""

SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SUPERVISOR_SYSTEM),
    ("placeholder", "{messages}"),
])

def supervisor_node(state: AppointmentState) ->dict:
    messages = state["messages"]
    if messages and isinstance(messages[-1], HumanMessage):
        previous_assistant = next(
            (message for message in reversed(messages[:-1]) if isinstance(message, AIMessage)),
            None,
        )
        latest_user_text = messages[-1].content
        previous_assistant_text = (
            previous_assistant.content if previous_assistant is not None else ""
        )
        if (
            isinstance(latest_user_text, str)
            and isinstance(previous_assistant_text, str)
            and "reschedul" in previous_assistant_text.lower()
            and "doctor" in previous_assistant_text.lower()
            and "?" in previous_assistant_text
            and any(
                phrase in latest_user_text.lower()
                for phrase in ("book with ", "reschedule with ", "use dr. ", "use doctor ")
            )
        ):
            return {"intent": "reschedule", "next_agent": "rescheduling_agent"}

    llm =  model.with_structured_output(SupervisorDecision)

    chain = SUPERVISOR_PROMPT | llm
    decision : SupervisorDecision = chain.invoke(
        {"messages": sanitize_messages(state["messages"])}
    )
    return {
        "intent": decision.intent,
        "next_agent": decision.next_agent
    }
