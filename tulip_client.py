import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import requests
from dotenv import load_dotenv

# # --- Tulip instance + auth ---
# TULIP_BASE=https://th-deg.tulip.co/api/v3
# TULIP_AUTH_HEADER=Basic YXBpa2V5LjJfUzJtM0R0Z2J2UEhTZG1XQUQ6ZFQxZHY3NVRoQ2IyVGUyaGluTlY3T25Qdm5fX21FLTNMdGNyM3Uwb213Rw==

# # --- Tables you will touch ---
# TULIP_TABLE_UID=CDbvqYC4p8F2AbjiC

# # --- Machine + attributes  ---
# MACHINE_ID=De6x7Cq6jAr2uwvYb
# UID_ATTRIBUTE_ID=3ixY5Myhe6iYK9uBK
# TABLE_ID_ATTRIBUTE_ID=mTE76Tou79dCvsAk8

load_dotenv()

TULIP_BASE = os.getenv("TULIP_BASE", "").rstrip("/")
AUTH_HEADER = os.getenv("TULIP_AUTH_HEADER", "")
TULIP_TABLE_UID = os.getenv("TULIP_TABLE_UID", "")

MACHINE_ID = os.getenv("MACHINE_ID", "")
UID_ATTRIBUTE_ID = os.getenv("UID_ATTRIBUTE_ID", "")
TABLE_ID_ATTRIBUTE_ID = os.getenv("TABLE_ID_ATTRIBUTE_ID", "")

COMMON_HEADERS = {
    "Authorization": AUTH_HEADER,          # HTTP Basic with API token
    "Content-Type": "application/json",
}

def _utc_rfc3339(dt: datetime) -> str:
    """UTC ISO string with Z (RFC3339)."""
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

def _ms_between(start: datetime, end: datetime) -> int:
    """Tulip Interval fields are integers of milliseconds (e.g., 123000)."""
    return int((end - start).total_seconds() * 1000)

class TulipError(RuntimeError):
    pass

@dataclass
class TulipClient:
    base_url: str = TULIP_BASE
    headers: Dict[str, str] = None

    def __post_init__(self):
        if not self.base_url:
            raise TulipError("TULIP_BASE is not set")
        if not AUTH_HEADER:
            raise TulipError("TULIP_AUTH_HEADER is not set")
        self.headers = COMMON_HEADERS

    # ---------- Tables ----------
    def create_record(self, table_id: str, record: Dict[str, Any]) -> Dict[str, Any]:
        """POST /tables/{tableId}/records (record must include 'id')."""
        url = f"{self.base_url}/tables/{table_id}/records"
        r = requests.post(url, headers=self.headers, json=record, timeout=30)
        if r.status_code != 201:
            raise TulipError(f"Create failed ({r.status_code}): {r.text}")
        return r.json()

    def list_records(
        self,
        table_id: str,
        limit: int = 10,
        offset: int = 0,
        filters: Optional[List[Dict[str, Any]]] = None,
        sort_options: Optional[List[Dict[str, str]]] = None,
        fields: Optional[Sequence[str]] = None,
        include_total_count: bool = False,
    ) -> Dict[str, Any]:
        """GET /tables/{tableId}/records with JSON-encoded query params."""
        url = f"{self.base_url}/tables/{table_id}/records"
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if include_total_count:
            params["includeTotalCount"] = "true"
        if filters:
            params["filters"] = json.dumps(filters)
        if sort_options:
            params["sortOptions"] = json.dumps(sort_options)
        if fields:
            params["fields"] = json.dumps(list(fields))
        r = requests.get(url, headers=self.headers, params=params, timeout=30)
        if r.status_code != 200:
            raise TulipError(f"List failed ({r.status_code}): {r.text}")
        # X-Total-Count is present only if include_total_count=true
        return {"records": r.json(), "total": r.headers.get("X-Total-Count")}

    def get_record(self, table_id: str, record_id: str) -> Dict[str, Any]:
        """GET /tables/{tableId}/records/{id}"""
        url = f"{self.base_url}/tables/{table_id}/records/{record_id}"
        r = requests.get(url, headers=self.headers, timeout=30)
        if r.status_code != 200:
            raise TulipError(f"Get failed ({r.status_code}): {r.text}")
        return r.json()

    def update_record(self, table_id: str, record_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
        """PATCH /tables/{tableId}/records/{id} (partial update)."""
        url = f"{self.base_url}/tables/{table_id}/records/{record_id}"
        r = requests.patch(url, headers=self.headers, json=patch, timeout=30)
        if r.status_code not in (200, 204):
            raise TulipError(f"Update failed ({r.status_code}): {r.text}")
        return r.json() if r.content else {}

    # ---------- Machine Attributes ----------
    def report_attributes(self, attributes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        POST /attributes/report
        Body: { "attributes": [ { "machineId": "...", "attributeId": "...", "value": <typed> }, ... ] }
        """
        url = f"{self.base_url}/attributes/report"
        body = {"attributes": attributes}
        r = requests.post(url, headers=self.headers, json=body, timeout=30)
        if r.status_code not in (200, 204):
            raise TulipError(f"Attributes report failed ({r.status_code}): {r.text}")
        return r.json() if r.content else {}

# --------- Helpers specific to your inspection table schema ---------
# Your field UIDs (types in comments)
FIELDS = {
    "id": "id",  # (Text, must be unique)
    "station_id": "ghkud_station_id",  # Text
    "start_utc": "dsuud_start_utc",  # Datetime
    "end_utc": "qllhd_end_utc",  # Datetime
    "duration_ms": "eiync_duration",  # Interval (milliseconds)
    "final_model_label": "nrrga_final_model_label",  # Text
    "status_motor_cover_fixed": "qncks_statusmotorcoverfixed",  # boolean
    "status_function_of_roller": "ayliv_status_function_roller",  # boolean
    "status_function_of_cabin": "vwqlw_status_function_cabin",  # boolean
    "status_position_of_light": "snihp_status_position_light",  # boolean
    "top_light_mode": "fgcay_top_light_mode",  # text
    "status_tires_correct_assembled": "ystqx_status_tires_correct_assembled",  # boolean
    "front_tire_type": "epfss_front_tire_type",  # text
    "rear_tire_type": "syeaf_rear_tiretype",  # text
    "status_all_parts_attached": "pprfc_status_all_parts_attached",  # boolean
    "overall_result": "nfiho_overall_result",  # boolean
    "found_features": "xrezz_found_features",  # text
    "found_features_count": "txygq_found_features_count",  # integer  
    "missing_features": "aftja_missing_features",  # text
}

def build_record_from_domain(
    *,
    id_value: str,
    station_id: str,
    start_dt: datetime,
    end_dt: datetime,
    final_model_label: str,
    status_motor_cover_fixed: bool,
    status_function_of_roller: bool,
    status_function_of_cabin: bool,
    status_position_of_light: bool,
    top_light_mode: str,
    status_tires_correct_assembled: bool,
    front_tire_type: str,
    rear_tire_type: str,
    status_all_parts_attached: bool,
    overall_result: bool,
    found_features: Sequence[str],
    missing_features: Sequence[str],
) -> Dict[str, Any]:
    """
    Convert your domain fields into the Tulip table document with the correct UIDs and types.
    - Datetimes must be RFC3339 strings (UTC).
    - Interval expects milliseconds (int).
    - Booleans stay booleans; integers stay integers.
    """
    duration_ms = _ms_between(start_dt, end_dt)
    features_text = "; ".join(found_features)
    missing_text = "; ".join(missing_features)
    doc = {
        FIELDS["id"]: id_value,
        FIELDS["station_id"]: station_id,
        FIELDS["start_utc"]: _utc_rfc3339(start_dt),
        FIELDS["end_utc"]: _utc_rfc3339(end_dt),
        FIELDS["duration_ms"]: duration_ms,
        FIELDS["final_model_label"]: final_model_label,
        FIELDS["status_motor_cover_fixed"]: status_motor_cover_fixed,
        FIELDS["status_function_of_roller"]: status_function_of_roller,
        FIELDS["status_function_of_cabin"]: status_function_of_cabin,
        FIELDS["status_position_of_light"]: status_position_of_light,
        FIELDS["top_light_mode"]: top_light_mode,
        FIELDS["status_tires_correct_assembled"]: status_tires_correct_assembled,
        FIELDS["front_tire_type"]: front_tire_type,
        FIELDS["rear_tire_type"]: rear_tire_type,
        FIELDS["status_all_parts_attached"]: status_all_parts_attached,
        FIELDS["overall_result"]: overall_result,
        FIELDS["found_features"]: features_text,
        FIELDS["found_features_count"]: int(len(found_features)),
        FIELDS["missing_features"]: missing_text,
    }
    return doc
