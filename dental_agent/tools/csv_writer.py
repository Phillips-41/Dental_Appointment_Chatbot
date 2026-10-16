from typing import List, Dict, Any, Optional
import pandas as pd
from langchain_core.tools import tool

from dental_agent.config.setting import CSV_PATH
from dental_agent.utils import format_slot

def _load_df() -> pd.DataFrame:

    """
    Load and normalize the CSV data into a pandas DataFrame.
    """
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    df["is_available"] = df["is_available"].astype(str).str.upper() == "TRUE"

    df["date_slot"] = pd.to_datetime(df["date_slot"], format="mixed",dayfirst=False)

    df["doctor_name"]= df["doctor_name"].str.lower().str.strip()
    df["specialization"] = df["specialization"].str.lower().str.strip()

    df["patient_to_attend"] = (
        df["patient_to_attend"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .replace({"nan":"","None":""})
    )
    return df

def _save_df(df: pd.DataFrame) -> None:
    """
    Write the DataFrame back to the CSV file.
    """
    out = df.copy()
    out["date_slot"] = out["date_slot"].apply(format_slot)
    out["is_available"] = out["is_available"].map({True: "TRUE", False: "FALSE"})
    out["patient_to_attend"] = out["patient_to_attend"].replace("nan","")
    out.to_csv(CSV_PATH, index=False)

@tool
def book_appointment(patient_id: str, doctor_name: str, date_slot: str)-> Dict[str, Any]:
    """
    Book an available slot for a patient.
    
    Return :
       {"success" : bool, "message":str}
    """
    df = _load_df()

    try:
        target_dt = pd.to_datetime(date_slot, format="mixed", dayfirst=False)
    except Exception as e:
        return {"success": False, "message": f"Invalid date format: {date_slot}"}
    doc = doctor_name.lower().strip()
    pid = str(patient_id).strip()

    mask = (df["doctor_name"] == doc) & (df["date_slot"] == target_dt)
    rows = df.loc[mask]

    if rows.empty:
        return {"success": False, "message": "Slot not available for this doctor"}
    if not rows.iloc[0]["is_available"]:
        return {"success": False, "message": "Slot is already booked"}
    df.loc[mask, "is_available"] = False
    df.loc[mask, "patient_to_attend"]= pid
    _save_df(df)
    return {
        "success": True,
        "message": f"Appointment booked for patient {pid} with Dr. {doctor_name} on {format_slot(target_dt)}"   
    }


@tool
def reschedule_appointment(
    patient_id: str,
    current_date_slot: str,
    new_date_slot: str,
    doctor_name: str,
) -> Dict[str, Any]:
    """
    Atomically move a patient from one slot to another (same doctor).
    """
    df = _load_df()

    try:
        current_dt = pd.to_datetime(current_date_slot, format="mixed", dayfirst=False)
        new_dt = pd.to_datetime(new_date_slot, format="mixed", dayfirst=False)
    except Exception as e:
        return {"success": False, "message": f"Date parse error: {e}"}

    doc = doctor_name.lower().strip()
    pid = str(patient_id).strip()

    # 1. Verify old booking exists
    old_mask = (
        (df["patient_to_attend"] == pid)
        & (df["date_slot"] == current_dt)
        & (~df["is_available"])
    )
    if df.loc[old_mask].empty:
        return {
            "success": False,
            "message": f"No existing booking for patient {pid} at {current_date_slot}.",
        }

    # 2. Verify new slot is free
    new_mask = (df["doctor_name"] == doc) & (df["date_slot"] == new_dt)
    new_rows = df.loc[new_mask]
    if new_rows.empty:
        return {"success": False, "message": f"New slot {new_date_slot} does not exist."}
    if not new_rows.iloc[0]["is_available"]:
        return {"success": False, "message": f"New slot {new_date_slot} is already taken."}

    # 3. Perform the move
    df.loc[old_mask, "is_available"] = True
    df.loc[old_mask, "patient_to_attend"] = ""
    df.loc[new_mask, "is_available"] = False
    df.loc[new_mask, "patient_to_attend"] = pid
    _save_df(df)

    return {
        "success": True,
        "message": (
            f"Rescheduled patient {pid} from {current_date_slot} "
            f"to {new_date_slot} with {doctor_name}."
        ),
    }

@tool
def cancel_appointment(patient_id: str, date_slot: str) -> Dict[str, Any]:
    """
    Cancel a patient's appointment and free the slot.
    """
    df = _load_df()

    try:
        target_dt = pd.to_datetime(date_slot, format="mixed", dayfirst=False)
    except Exception:
        return {"success": False, "message": f"Invalid date format: {date_slot}"}

    pid = str(patient_id).strip()
    mask = (
        (df["patient_to_attend"] == pid)
        & (df["date_slot"] == target_dt)
        & (~df["is_available"])
    )

    if df.loc[mask].empty:
        return {
            "success": False,
            "message": f"No booking found for patient {pid} at {date_slot}.",
        }

    df.loc[mask, "is_available"] = True
    df.loc[mask, "patient_to_attend"] = ""
    _save_df(df)

    return {
        "success": True,
        "message": f"Cancelled appointment for patient {pid} at {date_slot}.",
    }
