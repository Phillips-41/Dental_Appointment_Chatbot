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





@tool
def list_doctors_by_specialization(specialization: str) -> list:
    """
    Return distinct doctor names for a given specialization.

    Args:
        specialization: e.g. 'orthodontist'.

    Returns:
        Sorted list of doctor name strings.
    """
    df = _load_df()
    mask = df["specialization"] == specialization.lower().strip()
    return sorted(df[mask]["doctor_name"].unique().tolist())


@tool
def check_slot_availability(doctor_name: str, date_slot: str) -> dict:
    """
    Check if a specific doctor slot is available.

    Args:
        doctor_name: Doctor name, e.g. 'emily johnson'.
        date_slot: Slot string in M/D/YYYY H:MM format, e.g. '5/10/2026 9:00'.

    Returns:
        Dict with keys: found (bool), is_available (bool), patient_to_attend (str).
    """
    df = _load_df()
    try:
        target_dt = pd.to_datetime(date_slot, format="mixed", dayfirst=False)
    except Exception:
        return {"found": False, "is_available": False, "patient_to_attend": ""}

    mask = (df["doctor_name"] == doctor_name.lower().strip()) & (df["date_slot"] == target_dt)
    rows = df[mask]

    if rows.empty:
        return {"found": False, "is_available": False, "patient_to_attend": ""}

    row = rows.iloc[0]
    return {
        "found": True,
        "is_available": bool(row["is_available"]),
        "patient_to_attend": row["patient_to_attend"],
    }


@tool
def get_available_slots(
    specialization: str= "",
    doctor_name: str="",
    date_filter: str=""
)-> list:
    """
    Return available (is_available=True) appointment slots.
    
    Args:
        specializtion: Filter by specialization, e.g. 'orthodontist'. Leave empty to skip.
        doctor_name: Filter by doctor name, e.g. 'emily johnson'. Leave empty to skip.
        date_filter: Filter by date string M/D/YYYY, e.g. '5/10/2026'. Leave empty to skip.

    Returns:
        List of dicts with keys: date_slot, specialization, doctor_name.
        Returns at most 20 rows to keep response concise.
    """

    df = _load_df()
    mask = df["is_available"]

    if specialization:
        mask = mask & (df["specialization"] == specialization.lower().strip())
    if doctor_name:
        mask = mask & (df["doctor_name"] == doctor_name.lower().strip())
    if date_filter:
        try:
            target_date = pd.to_datetime(date_filter).date()
            mask = mask & (df["date_slot"].dt.date == target_date)
        except Exception:
            pass

    result = df[mask][["date_slot", "specialization", "doctor_name"]].copy()
    result["date_slot"] = result["date_slot"].apply(format_slot)
    return result.head(20).to_dict(orient="records")


@tool
def get_patient_appointments(patient_id: str) -> list:
    """
    Return all booked appointments for a given patient ID.

    Args:
        patient_id: Numeric patient ID string, e.g. '1000082'.

    Returns:
        List of dicts with keys: date_slot, specialization, doctor_name, patient_to_attend.
    """
    df = _load_df()
    mask = df["patient_to_attend"] == str(patient_id).strip()
    result = df[mask][["date_slot", "specialization", "doctor_name", "patient_to_attend"]].copy()
    result["date_slot"] = result["date_slot"].apply(format_slot)
    return result.to_dict(orient="records")