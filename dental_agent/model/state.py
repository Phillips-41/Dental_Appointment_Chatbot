from typing import Annotated, List, Optional, Literal, TypedDict
from langchain_core.messages import BaseMessage
import operator

IntentType = Literal["get_info", "book", "cancel", "reschedule","unknown", "end"]

RouterTarget = Literal["info_agent", "booking_agent", "cancellation_agent", "rescheduling_agent","end"]

class AppointmentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]

    intent: Optional[IntentType]
    next_agent: Optional[RouterTarget]

    patient_id: Optional[str]
    requested_specialization: Optional[str]
    requested_doctor: Optional[str]
    requested_date_slot: Optional[str]

    current_date_slot: Optional[str]
    new_date_slot: Optional[str]

    available_slots: Optional[List[dict]]
    operation_success: Optional[bool]
    operation_message: Optional[str]

    final_response: Optional[str]