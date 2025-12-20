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
from typing import Any, Dict, List, Optional

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
# Integrity checks (draft/roster)
# ---------------------------------------------------------------------------

class DraftIntegrityError(RuntimeError):
    """Raised when the draft state is internally inconsistent."""

    def __init__(self, stage: str, issues: List[str]):
        self.stage = stage
        self.issues = issues
        msg = f"Draft integrity check failed at '{stage}':\n- " + "\n- ".join(issues)
        super().__init__(msg)


def validate_roster_integrity(
    roster_df=None,
    required_fields: Optional[List[str]] = None,
) -> List[str]:
    """Validate basic assumptions about the roster table.

    Returns a list of human-readable issues (empty if OK).
    """
    issues: List[str] = []
    df = roster_df if roster_df is not None else ROSTER_DF
    req = required_fields if required_fields is not None else MIN_BASE_FIELDS

    cols = list(getattr(df, "columns", []))
    for col in req:
        if col not in cols:
            issues.append(f"ROSTER_DF missing required column: {col}")

    if "PlayerID" not in cols:
        issues.append("ROSTER_DF missing 'PlayerID' column (expected for draft inserts).")
        return issues

    try:
        if len(df) > 0:
            sample = df.head(20)
            for idx, pid in zip(list(sample.index), list(sample["PlayerID"])):
                if int(idx) != int(pid):
                    issues.append("ROSTER_DF index does not match PlayerID column (expected index=PlayerID).")
                    break
    except Exception as e:
        issues.append(f"Failed to validate roster index/PlayerID: {e}")

    return issues


def validate_draft_integrity(
    draft_state: Optional[Dict[str, Any]] = None,
    roster_df=None,
    *,
    strict: bool = True,
    stage: str = "unknown",
) -> Dict[str, Any]:
    """Validate draft state invariants.

    - If strict=True, raises DraftIntegrityError when issues exist.
    - Returns {"ok": bool, "issues": [...], "stage": stage}.
    """
    issues: List[str] = []

    ds = draft_state if draft_state is not None else _ensure_draft_state()
    df = roster_df if roster_df is not None else ROSTER_DF

    allowed_phases = {"idle", "class_generated", "lottery_done", "draft_in_progress", "draft_completed"}
    phase = ds.get("phase")
    if phase not in allowed_phases:
        issues.append(f"Invalid phase: {phase}")

    prospects = ds.get("prospects")
    board = ds.get("board")
    if not isinstance(prospects, dict):
        issues.append("draft.prospects must be a dict")
        prospects = {}
    if not isinstance(board, list):
        issues.append("draft.board must be a list")
        board = []

    seen_ids = set()
    for k, p in prospects.items():
        try:
            pid = int(p.get("id"))
        except Exception:
            issues.append(f"Prospect '{k}' has invalid id: {p.get('id')}")
            continue

        if str(pid) != str(k):
            issues.append(f"Prospect key '{k}' does not match its id '{pid}'")

        if pid in seen_ids:
            issues.append(f"Duplicate prospect id detected: {pid}")
        seen_ids.add(pid)

        if "true_ratings" not in p:
            issues.append(f"Prospect {pid} missing true_ratings (should exist internally).")

        drafted_by = p.get("drafted_by")
        draft_pick = p.get("draft_pick")
        if drafted_by:
            try:
                int(draft_pick)
            except Exception:
                issues.append(f"Prospect {pid} drafted_by set but draft_pick invalid: {draft_pick}")

    if len(board) != len(set(board)):
        issues.append("draft.board contains duplicate prospect ids.")
    for pid in board:
        if str(pid) not in prospects:
            issues.append(f"draft.board references missing prospect id: {pid}")

    lot = ds.get("lottery", {})
    order_round1 = lot.get("draft_order_round1", [])
    if order_round1 and (not isinstance(order_round1, list) or len(order_round1) < 30):
        issues.append("lottery.draft_order_round1 must be a list of 30 team_ids when set.")

    d = ds.get("draft", {})
    order = d.get("order", [])
    if order and (not isinstance(order, list) or len(order) != 30):
        issues.append("draft.order must be a list of length 30 when set.")

    try:
        cur_idx = int(d.get("current_pick_index", 0))
        if cur_idx < 0 or cur_idx > 30:
            issues.append(f"draft.current_pick_index out of range: {cur_idx}")
    except Exception:
        issues.append(f"draft.current_pick_index not int-like: {d.get('current_pick_index')}")

    picks = d.get("picks", [])
    if picks and not isinstance(picks, list):
        issues.append("draft.picks must be a list.")
        picks = []

    for rec in picks:
        try:
            pick_no = int(rec.get("pick_no"))
            team_id = rec.get("team_id")
            prospect_id = int(rec.get("prospect_id"))
            player_id = int(rec.get("player_id"))
        except Exception:
            issues.append(f"Malformed pick record: {rec}")
            continue

        p = prospects.get(str(prospect_id))
        if not p:
            issues.append(f"Pick {pick_no} references missing prospect_id {prospect_id}")
            continue

        if str(p.get("drafted_by")) != str(team_id):
            issues.append(
                f"Pick {pick_no}: prospect {prospect_id} drafted_by mismatch ({p.get('drafted_by')} vs {team_id})"
            )

        if int(p.get("id")) != int(player_id):
            issues.append(f"Pick {pick_no}: player_id mismatch (prospect id {p.get('id')} vs record {player_id})")

        try:
            if player_id not in df.index:
                issues.append(f"Pick {pick_no}: player_id {player_id} not found in roster index")
        except Exception:
            issues.append(f"Pick {pick_no}: could not verify roster contains player_id {player_id}")

    issues.extend(validate_roster_integrity(df))

    try:
        nxt = int(GAME_STATE.get("next_player_id", 0) or 0)
        if len(df.index) > 0:
            mx = int(df.index.max())
            if nxt <= mx:
                issues.append(f"next_player_id ({nxt}) should be > max roster PlayerID ({mx}).")
    except Exception:
        pass

    ok = len(issues) == 0
    if (not ok) and strict:
        raise DraftIntegrityError(stage=stage, issues=issues)
    return {"ok": ok, "issues": issues, "stage": stage}
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



