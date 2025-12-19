from __future__ import annotations

"""draft_state.py — Draft state + shared constants/helpers

Split out of the original draft.py for easier maintenance.

- Draft state lives in state.GAME_STATE["draft"] (JSON-friendly dict only).
- This module owns:
  * rating column names / schema constants
  * small utility helpers (_utc_iso, _safe_int, etc.)
  * draft state initializer/reset
"""

from datetime import datetime
from typing import Any, Dict

from state import GAME_STATE
from config import ROSTER_DF

# ---------------------------------------------------------------------------
# 0) 능력치 스키마(컬럼 이름은 사용자가 정의한 형태를 그대로 사용)
# ---------------------------------------------------------------------------

# 하위(세부) 능력치 컬럼들
OUTSIDE_SUB = [
    "Close Shot",
    "Mid-Range Shot",
    "Three-Point Shot",
    "Free Throw",
    "Shot IQ",
    "Offensive Consistency",
]
INSIDE_SUB = [
    "Layup",
    "Standing Dunk",
    "Driving Dunk",
    "Post Hook",
    "Post Fade",
    "Post Control",
    "Draw Foul",
    "Hands",
]
PLAYMAKING_SUB = [
    "Pass Accuracy",
    "Ball Handle",
    "Speed with Ball",
    "Pass IQ",
    "Pass Vision",
]
DEFENSE_SUB = [
    "Interior Defense",
    "Perimeter Defense",
    "Steal",
    "Block",
    "Help Defense IQ",
    "Pass Perception",
    "Defensive Consistency",
]
REBOUNDING_SUB = [
    "Offensive Rebound",
    "Defensive Rebound",
]
ATHLETICISM_SUB = [
    "Speed",
    "Agility",
    "Strength",
    "Vertical",
    "Stamina",
    "Hustle",
]

# 그룹/요약 능력치 컬럼들
GROUP_COLS = [
    "Outside Scoring",
    "Inside Scoring",
    "Playmaking",
    "Defense",
    "Rebounding",
    "Athleticism",
]
META_COLS = [
    "OVR",
    "Overall Durability",
    "Potential",
]

# 드래프트에서 우리가 "필수"로 채워 넣는 최소 기본 필드
MIN_BASE_FIELDS = ["Name", "Team", "POS", "Age", "OVR", "Potential"]

# 가능하면 로스터에 같이 넣고 싶은(있을 때만 채움) 바이오 필드 후보
BIO_FIELD_CANDIDATES = [
    "Height",
    "Height (in)",
    "Height W/O Shoes",
    "Wingspan",
    "Wingspan (in)",
    "Weight",
    "Weight (lbs)",
    "College",
    "Nationality",
    "Years Pro",
]

ALL_SUB_RATINGS = OUTSIDE_SUB + INSIDE_SUB + PLAYMAKING_SUB + DEFENSE_SUB + REBOUNDING_SUB + ATHLETICISM_SUB
ALL_RATING_COLS = GROUP_COLS + META_COLS + ALL_SUB_RATINGS


# ---------------------------------------------------------------------------
# PlayerID allocation (centralized)
# ---------------------------------------------------------------------------

def _max_numeric_roster_player_id() -> int:
    """Find the current max numeric PlayerID from ROSTER_DF.index.

    We keep this as a fallback initializer for GAME_STATE["next_player_id"].
    """
    try:
        # Fast path if index is numeric-ish
        if len(ROSTER_DF.index) == 0:
            return 0
        return int(max(ROSTER_DF.index))
    except Exception:
        mx = 0
        for i in list(getattr(ROSTER_DF, "index", [])):
            try:
                mx = max(mx, int(i))
            except Exception:
                continue
        return mx

def ensure_next_player_id() -> int:
   """Ensure GAME_STATE has a monotonically increasing next_player_id counter."""
    try:
        cur = GAME_STATE.get("next_player_id", None)
        if cur is None:
            GAME_STATE["next_player_id"] = _max_numeric_roster_player_id() + 1
        else:
            GAME_STATE["next_player_id"] = int(cur)
    except Exception:
        GAME_STATE["next_player_id"] = _max_numeric_roster_player_id() + 1
    return int(GAME_STATE["next_player_id"])

def reserve_player_ids(count: int) -> int:
   """Reserve a contiguous block of player IDs and return the first ID."""
    if count <= 0:
        raise ValueError("count must be > 0")
    start = ensure_next_player_id()
    GAME_STATE["next_player_id"] = start + int(count)
    return start


# ---------------------------------------------------------------------------
# 1) Draft State Helpers
# ---------------------------------------------------------------------------

def _utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _clamp(v: float, lo: float = 25.0, hi: float = 99.0) -> float:
    return max(lo, min(hi, v))


def _round1(v: float) -> float:
    # UI 표시를 고려해 소수 1자리 정도
    return round(float(v), 1)


def _ensure_draft_state() -> Dict[str, Any]:
    """
    GAME_STATE["draft"] 구조를 보장하고 반환한다.
    phase:
      - idle
      - class_generated (prospects created; scouting available)
      - lottery_done
      - draft_in_progress
      - draft_completed
    """
    draft = GAME_STATE.setdefault("draft", {})
    draft.setdefault("schema_version", "1.0")
    draft.setdefault("phase", "idle")
    draft.setdefault("season_id", None)
    draft.setdefault("created_at", None)
    draft.setdefault("rng_seed", None)

    draft.setdefault("prospects", {})  # prospect_id(str) -> dict (json-friendly)
    draft.setdefault("board", [])      # prospect_id list (initial ordering)
    draft.setdefault("scouting", {
        "my_team_id": None,
        "max_requests": 10,
        "requested": [],          # prospect_id list
        "combine_revealed": [],    # prospect_id list
        "workout_revealed": [],    # prospect_id list
    })
    draft.setdefault("lottery", {
        "standings_worst_to_best": [],
        "draft_order_round1": [],
        "top4": [],
        "weights_1st_pick": [],
        "ran_at": None,
    })
    draft.setdefault("draft", {
        "round": 1,
        "current_pick_index": 0,  # 0-based
        "order": [],              # team_id per pick (length 30)
        "picks": [],              # list of {pick_no, team_id, prospect_id, ts}
        "completed": False,
        "started_at": None,
        "completed_at": None,
    })
    return draft


def reset_draft_state() -> Dict[str, Any]:
    """드래프트 관련 상태를 초기화한다(루키 클래스/로터리/드래프트 진행 모두 삭제)."""
    GAME_STATE["draft"] = {}
    return _ensure_draft_state()



