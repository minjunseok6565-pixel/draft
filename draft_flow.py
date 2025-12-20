from __future__ import annotations

"""draft_flow.py — Scouting requests + lottery + draft execution

Split out of the original draft.py.

This module owns:
- request_evaluation()
- lottery helpers (set_standings_and_run_lottery, etc.)
- draft execution (start_draft_if_ready, make_pick, auto_pick_current_team, advance_until_user_pick)
- optional helper maybe_initialize_after_playoffs()

Note: _hash_seed uses a stable hash (crc32) so results stay the same after game restart.
"""

from typing import Any, Dict, List, Optional
import random
import zlib

from config import ROSTER_DF
from state import GAME_STATE

from draft_state import (
    _ensure_draft_state,
    _utc_iso,
    ALL_RATING_COLS,
    BIO_FIELD_CANDIDATES,
    MIN_BASE_FIELDS,
    validate_draft_integrity,
)

from rookies import (
    _public_prospect_view,
    _combine_metrics,
    _workout_report,
    get_public_prospect_board,
    initialize_rookie_class_if_needed,
)

# ---------------------------------------------------------------------------
# 9) Scouting Requests: Combine & Workout (최대 10명)
# ---------------------------------------------------------------------------

def request_evaluation(
    prospect_id: int,
    reveal_combine: bool = True,
    reveal_workout: bool = True,
    my_team_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    유저가 특정 선수에 대해 "컴바인 + 워크아웃" 정보를 요청한다.
    - 기본적으로 reveal_combine/reveal_workout 둘 다 True
    - 최대 요청 가능 인원은 draft.scouting.max_requests (기본 10)
    """
    draft = _ensure_draft_state()
    scouting = draft.get("scouting", {})
    if my_team_id:
        scouting["my_team_id"] = my_team_id
    my_team_id = scouting.get("my_team_id")

    if not my_team_id:
        raise ValueError("my_team_id is not set. Provide my_team_id or set it in draft.scouting.")

    p = draft.get("prospects", {}).get(str(prospect_id))
    if not p:
        raise KeyError(f"Prospect not found: {prospect_id}")

    if p.get("drafted_by"):
        # 이미 지명된 선수면 추가 평가 의미 없음
        return {"prospect": _public_prospect_view(p), "remaining_requests": _remaining_requests(draft)}

    # 요청 한도 체크 (prospect 단위로 카운트)
    requested: List[int] = scouting.get("requested", [])
    if prospect_id not in requested:
        if len(requested) >= int(scouting.get("max_requests", 10)):
            raise ValueError("No remaining evaluation requests.")
        requested.append(prospect_id)
        scouting["requested"] = requested

    # combine 공개
    if reveal_combine and not p.get("reveals", {}).get("combine"):
        p["combine"] = _combine_metrics(random.Random(_hash_seed(draft, prospect_id, "combine")), p["pos"], p["bio"], p["true_ratings"])
        p["reveals"]["combine"] = True
        scouting.setdefault("combine_revealed", [])
        if prospect_id not in scouting["combine_revealed"]:
            scouting["combine_revealed"].append(prospect_id)

    # workout 공개
    if reveal_workout and not p.get("reveals", {}).get("workout"):
        pub = _public_prospect_view(p)
        p["workout_report"] = _workout_report(random.Random(_hash_seed(draft, prospect_id, "workout")), my_team_id, pub, p["true_ratings"])
        p["reveals"]["workout"] = True
        scouting.setdefault("workout_revealed", [])
        if prospect_id not in scouting["workout_revealed"]:
            scouting["workout_revealed"].append(prospect_id)

    return {"prospect": _public_prospect_view(p), "remaining_requests": _remaining_requests(draft)}


def _remaining_requests(draft: Dict[str, Any]) -> int:
    scouting = draft.get("scouting", {})
    maxr = int(scouting.get("max_requests", 10))
    used = len(scouting.get("requested", []))
    return max(0, maxr - used)


def _hash_seed(draft: Dict[str, Any], pid: int, tag: str) -> int:
    """Stable seed for scouting outputs.

    - 같은 시즌/seed + 같은 선수 + 같은 tag("combine"/"workout")면 항상 같은 결과를 내기 위해 쓰는 시드.
    - Python의 hash()는 재실행마다 바뀔 수 있어서, 안정적인 crc32를 사용한다.
    """
    base = int(draft.get("rng_seed") or 1234567)
    tag_h = zlib.crc32(tag.encode("utf-8")) & 0xFFFFFFFF
    x = (base * 1000003) ^ (pid * 9176) ^ tag_h
    return x & 0x7FFFFFFF


# ---------------------------------------------------------------------------
# 10) Draft Lottery (하위 14팀)
# ---------------------------------------------------------------------------

LOTTERY_WEIGHTS_1ST_PICK = [
    ("14th_worst", 0.5),
    ("13th_worst", 1.0),
    ("12th_worst", 1.5),
    ("11th_worst", 2.0),
    ("10th_worst", 3.0),
    ("9th_worst", 4.5),
    ("8th_worst", 6.0),
    ("7th_worst", 7.5),
    ("6th_worst", 9.0),
    ("5th_worst", 10.5),
    ("4th_worst", 12.5),
    ("3rd_worst", 14.0),
    ("2nd_worst", 14.0),
    ("worst", 14.0),
]
# 위는 라벨용. 실제 사용은 입력 standings 순서(worst->best of 14)에 맞춰 가중치를 배정한다.
LOTTERY_1ST_PICK_PCTS = [14.0, 14.0, 14.0, 12.5, 10.5, 9.0, 7.5, 6.0, 4.5, 3.0, 2.0, 1.5, 1.0, 0.5]


def run_draft_lottery(
    standings_worst_to_best: List[str],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    하위 14팀에 대해 로터리 추첨을 실행하여 픽 1~14 순서를 만든다.
    - 입력: standings_worst_to_best 는 전체 30팀 중 '정규시즌 성적' 기준 worst->best 순서여야 하며,
      이 함수는 그 중 앞 14개를 로터리 대상으로 사용한다.
    - 출력: round1_order (length 14), top4 (length 4)

    구현 방식(단순 근사):
    1) 1픽: 확률(1픽 확률)로 샘플링
    2) 2~4픽: 남은 팀에서 동일한 가중치로 반복 샘플링
    3) 5~14픽: 남은 팀을 원래 성적 순(worst->best)대로 배치
    """
    if len(standings_worst_to_best) < 14:
        raise ValueError("Need at least 14 teams for lottery standings.")

    bottom14 = list(standings_worst_to_best[:14])
    weights = list(LOTTERY_1ST_PICK_PCTS)

    rng = random.Random(seed if seed is not None else random.randint(1, 2_000_000_000))

    remaining = bottom14[:]
    remaining_weights = weights[:]

    top4: List[str] = []
    for _ in range(4):
        # weighted sample 1
        total = sum(remaining_weights)
        r = rng.random() * total
        acc = 0.0
        chosen_idx = 0
        for i, w in enumerate(remaining_weights):
            acc += w
            if r <= acc:
                chosen_idx = i
                break
        chosen_team = remaining.pop(chosen_idx)
        remaining_weights.pop(chosen_idx)
        top4.append(chosen_team)

    # picks 5-14: remaining in original worst->best order (as they appear in bottom14)
    # but we removed some, so filter bottom14 by remaining set
    rem_set = set(remaining)
    rest = [t for t in bottom14 if t in rem_set]

    order14 = top4 + rest
    return {
        "top4": top4,
        "order_1_to_14": order14,
        "weights_1st_pick": dict(zip(bottom14, weights)),
    }


def build_full_draft_order_round1(
    standings_worst_to_best_all30: List[str],
    seed: Optional[int] = None,
) -> List[str]:
    """
    전체 30팀의 정규시즌 성적(worst->best) 리스트를 받아
    1라운드 픽 순서(30개)를 반환한다.
    """
    if len(standings_worst_to_best_all30) != 30:
        raise ValueError("Expected standings list for all 30 teams (worst->best).")

    lottery = run_draft_lottery(standings_worst_to_best_all30[:14], seed=seed)
    order_1_14 = lottery["order_1_to_14"]
    # 15~30: 나머지 팀들(플레이오프 팀들)을 성적 역순(=worst->best among remaining)으로
    order_15_30 = standings_worst_to_best_all30[14:]
    return order_1_14 + order_15_30


def set_standings_and_run_lottery(
    standings_worst_to_best_all30: List[str],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    드래프트 상태에 standings를 저장하고 로터리를 실행한다.
    """
    draft = _ensure_draft_state()
    if not draft.get("prospects"):
        raise ValueError("Rookie class is not initialized. Call initialize_rookie_class_if_needed first.")

    rng_seed = seed if seed is not None else int(draft.get("rng_seed") or random.randint(1, 2_000_000_000))
    order = build_full_draft_order_round1(standings_worst_to_best_all30, seed=rng_seed)

    lot = draft.get("lottery", {})
    lot["standings_worst_to_best"] = standings_worst_to_best_all30
    lot["draft_order_round1"] = order
    lot["top4"] = order[:4]
    lot["ran_at"] = _utc_iso()
    lot["weights_1st_pick"] = [float(x) for x in LOTTERY_1ST_PICK_PCTS]
    draft["lottery"] = lot

    # draft.order도 같이 준비해두면 편함
    draft["draft"]["order"] = order[:30]
    draft["phase"] = "lottery_done"
    validate_draft_integrity(stage="after_lottery_done", strict=True)
    return get_public_prospect_board()


# ---------------------------------------------------------------------------
# 11) Draft 진행 (1라운드 30픽)
# ---------------------------------------------------------------------------

def start_draft_if_ready() -> Dict[str, Any]:
    """
    로터리까지 끝난 상태에서 드래프트를 시작한다.
    """
    draft = _ensure_draft_state()
    if not draft.get("prospects"):
        raise ValueError("Rookie class not ready.")
    if not draft.get("draft", {}).get("order"):
        # 로터리를 아직 안 했으면 standings가 없다는 의미
        raise ValueError("Draft order is empty. Run lottery first (set_standings_and_run_lottery).")

    d = draft["draft"]
    d["started_at"] = d.get("started_at") or _utc_iso()
    d["completed"] = False
    draft["phase"] = "draft_in_progress"
    validate_draft_integrity(stage="after_start_draft", strict=True)
    return get_public_prospect_board()


def _format_salary(amount: float) -> str:
    # "$1,234,567"
    amt = int(round(max(0.0, amount)))
    return f"${amt:,}"


def _rookie_salary_by_pick(pick_no: int) -> float:
    """
    단순 rookie scale(대략): 1픽 ~ 30픽
    실제 NBA와 다르지만 게임 밸런스용.
    """
    # 1픽: 11M, 30픽: 1.2M 정도
    hi, lo = 11_000_000.0, 1_200_000.0
    t = (pick_no - 1) / 29.0
    return hi * (1 - t) + lo * t


def _add_player_to_roster_df_inplace(player_row: Dict[str, Any]) -> None:
    """
    ROSTER_DF에 in-place로 행 추가.
    - player_row의 key는 ROSTER_DF.columns를 최대한 맞춘다.
    """
    # 누락 컬럼은 None으로 채우기
    row = {col: None for col in list(ROSTER_DF.columns)}
    for k, v in player_row.items():
        if k in row:
            row[k] = v
    # index는 이미 player_id로 결정되어 있다고 가정
    player_id = player_row.get("PlayerID")
    if player_id is None:
        # index를 player_id로 쓰는 기존 패턴을 따라간다.
        raise ValueError("player_row must include PlayerID for roster insertion.")
    ROSTER_DF.loc[int(player_id)] = row


def make_pick(
    team_id: str,
    prospect_id: int,
    enforce_turn: bool = True,
) -> Dict[str, Any]:
    """
    안전장치:
    - 중간에 에러가 나면(로스터 삽입 실패 등) 드래프트 상태가 '반쯤만' 적용되지 않도록
      가능한 범위에서 롤백(되돌리기)한다.
    """
    draft = _ensure_draft_state()
    d = draft.get("draft", {})
    order: List[str] = d.get("order") or []
    if not order:
        raise ValueError("Draft order is empty. Run lottery first.")
    if d.get("completed"):
        raise ValueError("Draft already completed.")

    order = d["draft_order_round1"]
    pick_idx = int(d.get("current_pick_index", 0))
    if pick_idx < 0 or pick_idx >= 30:
        raise ValueError("Invalid pick index.")

    current_team = order[pick_idx]
    if enforce_turn and current_team != team_id:
        raise ValueError(f"Not {team_id}'s turn. Current team is {current_team} (pick {pick_idx+1}).")

    p = draft.get("prospects", {}).get(str(prospect_id))
    if not p:
        raise KeyError(f"Prospect not found: {prospect_id}")
    if p.get("drafted_by"):
        raise ValueError("Prospect already drafted.")
    if "true_ratings" not in p:
        raise ValueError("Prospect missing true_ratings.")

    
    pick_no = pick_idx + 1

    true_r = p["true_ratings"]

    salary_amt = _rookie_salary_by_pick(pick_no)
    salary_str = _format_salary(salary_amt)

    # rookies.py에서 이미 예약된 player_id를 사용
    player_id = int(p["id"])
    if player_id in list(getattr(ROSTER_DF, "index", [])):
        raise ValueError(f"PlayerID collision: {player_id} already exists in roster.")

    player_row: Dict[str, Any] = {}
    player_row["PlayerID"] = player_id  # internal helper (insert uses it)
    player_row["Name"] = p["name"]
    player_row["Team"] = team_id
    player_row["POS"] = p["pos"]
    player_row["Age"] = p["age"]

    # 능력치 컬럼 채우기
    for col in ALL_RATING_COLS:
        if col in true_r:
            player_row[col] = true_r[col]

    # salary / salaryAmount
    if "Salary" in list(ROSTER_DF.columns):
        player_row["Salary"] = salary_str
    if "SalaryAmount" in list(ROSTER_DF.columns):
        player_row["SalaryAmount"] = float(salary_amt)

    # 바이오 컬럼이 있으면 채우기(있을 때만)
    bio = p.get("bio", {})
    for col in BIO_FIELD_CANDIDATES:
        if col not in list(ROSTER_DF.columns):
            continue
        if col in {"Height (in)", "Height W/O Shoes", "Height"}:
            player_row[col] = bio.get("height_in")
        elif col in {"Wingspan (in)", "Wingspan"}:
            player_row[col] = bio.get("wingspan_in")
        elif col in {"Weight", "Weight (lbs)"}:
            player_row[col] = bio.get("weight_lbs")
        elif col == "College":
            player_row[col] = p.get("college")
        elif col == "Years Pro":
            player_row[col] = 0

    # 필수 필드 검사(상업용 안전장치)
    missing = [k for k in MIN_BASE_FIELDS if (k not in player_row or player_row.get(k) is None)]
    if missing:
        raise ValueError(f"Cannot add rookie to roster: missing required fields {missing}")

    pick_record = {
        "pick_no": pick_no,
        "team_id": team_id,
        "player_id": player_id,
        "name": p.get("name"),
        "pos": p.get("pos"),
        "prospect_id": prospect_id,
        "ts": _utc_iso(),
    }

    # --- Transaction-like apply with rollback ---
    prev_phase = draft.get("phase")
    prev_pick_index = d.get("current_pick_index")
    prev_completed = d.get("completed")
    prev_completed_at = d.get("completed_at")
    prev_picks_len = len(d.get("picks", []) or [])

    prev_drafted_by = p.get("drafted_by")
    prev_draft_pick = p.get("draft_pick")

    players_meta = GAME_STATE.setdefault("players", {})
    had_player_meta = str(player_id) in players_meta
    prev_player_meta = players_meta.get(str(player_id))

    roster_inserted = False
    try:
        # 1) prospect mark drafted
        p["drafted_by"] = team_id
        p["draft_pick"] = pick_no

        # 2) roster insert (가장 실패 가능성이 큼)
        _add_player_to_roster_df_inplace(player_row)
        roster_inserted = True

        # 3) meta
        players_meta[str(player_id)] = {
            "rookie": True,
            "draft_season": draft.get("season_id"),
            "draft_pick": pick_no,
            "draft_team": team_id,
            "created_at": _utc_iso(),
        }

        # 4) log
        d.setdefault("picks", [])
        d["picks"].append(pick_record)

        # 5) advance pick (commit)
        d["current_pick_index"] = pick_idx + 1
        if d["current_pick_index"] >= 30:
            d["completed"] = True
            d["completed_at"] = _utc_iso()
            draft["phase"] = "draft_completed"

        draft["draft"] = d
        validate_draft_integrity(stage=f"after_make_pick_{pick_no}", strict=True)
        return get_public_prospect_board()

    except Exception:
        # rollback prospect fields
        if prev_drafted_by is None:
            p.pop("drafted_by", None)
        else:
            p["drafted_by"] = prev_drafted_by

        if prev_draft_pick is None:
            p.pop("draft_pick", None)
        else:
            p["draft_pick"] = prev_draft_pick

        # rollback roster row if inserted
        if roster_inserted:
            try:
                ROSTER_DF.drop(index=player_id, inplace=True, errors="ignore")
            except Exception:
                pass

        # rollback players meta
        if had_player_meta:
            players_meta[str(player_id)] = prev_player_meta
        else:
            players_meta.pop(str(player_id), None)

        # rollback picks list length
        try:
            if "picks" in d and isinstance(d["picks"], list):
                d["picks"] = d["picks"][:prev_picks_len]
        except Exception:
            pass

        # rollback pick cursor / completed flags / phase
        if prev_pick_index is None:
            d.pop("current_pick_index", None)
        else:
            d["current_pick_index"] = prev_pick_index

        if prev_completed is None:
            d.pop("completed", None)
        else:
            d["completed"] = prev_completed

        if prev_completed_at is None:
            d.pop("completed_at", None)
        else:
            d["completed_at"] = prev_completed_at

        draft["phase"] = prev_phase
        draft["draft"] = d
        raise

def auto_pick_current_team(strategy: str = "best_public") -> Dict[str, Any]:
    """
    AI 픽 자동 선택.
    - best_true_ovr: 남은 선수 중 true OVR 최대
    - best_public: 공개 보드 순서(대학 스탯 기반) 최상위
    """
    draft = _ensure_draft_state()
    d = draft.get("draft", {})
    order = d.get("order") or []
    if not order:
        raise ValueError("Draft order is empty.")
    if d.get("completed"):
        raise ValueError("Draft already completed.")
    pick_idx = int(d.get("current_pick_index", 0))
    team_id = order[pick_idx]

    prospects = draft.get("prospects", {})
    available = [p for p in prospects.values() if not p.get("drafted_by")]

    if not available:
        raise ValueError("No available prospects remaining.")

    if strategy == "best_public":
        # board 순서대로 첫 미지명
        for pid in draft.get("board", []):
            p = prospects.get(str(pid))
            if p and not p.get("drafted_by"):
                return make_pick(team_id, int(pid), enforce_turn=True)
        # fallback
        choice = random.choice(available)
        return make_pick(team_id, int(choice["id"]), enforce_turn=True)

    # 기본: true OVR 최대
    best = max(available, key=lambda x: float(x.get("true_ratings", {}).get("OVR", 0.0)))
    return make_pick(team_id, int(best["id"]), enforce_turn=True)


def advance_until_user_pick(
    my_team_id: str,
    auto_strategy: str = "best_true_ovr",
) -> Dict[str, Any]:
    """
    유저 팀 차례가 올 때까지 AI 픽을 자동 진행.
    - my_team_id 차례에 도달하면 멈추고 상태 반환
    """
    draft = _ensure_draft_state()
    d = draft.get("draft", {})
    order = d.get("order") or []
    if not order:
        raise ValueError("Draft order is empty.")
    if d.get("completed"):
        return get_public_prospect_board()

    while True:
        pick_idx = int(d.get("current_pick_index", 0))
        if pick_idx >= 30:
            break
        current_team = order[pick_idx]
        if current_team == my_team_id:
            break
        # AI 픽
        auto_pick_current_team(strategy=auto_strategy)
        d = draft.get("draft", {})

    return get_public_prospect_board()


# ---------------------------------------------------------------------------
# 12) "플레이오프 종료 후" 자동 준비 (선택)
# ---------------------------------------------------------------------------

def maybe_initialize_after_playoffs(
    my_team_id: Optional[str] = None,
    standings_worst_to_best_all30: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    플레이오프 종료(champion 존재) 시점에 루키 클래스/로터리를 자동으로 준비하는 헬퍼.
    - standings가 제공되면 바로 로터리까지 실행.
    - standings가 없으면 루키 클래스만 생성해 두고, 로터리는 나중에 실행.
    """
    draft = _ensure_draft_state()
    postseason = GAME_STATE.get("postseason") or {}
    champion = postseason.get("champion")
    if not champion:
        # 아직 플레이오프 종료 전
        return get_public_prospect_board()

    initialize_rookie_class_if_needed(my_team_id=my_team_id, seed=seed, force=False)

    if standings_worst_to_best_all30:
        set_standings_and_run_lottery(standings_worst_to_best_all30, seed=seed)

    return get_public_prospect_board()

