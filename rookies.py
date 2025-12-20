from __future__ import annotations

"""rookies.py — Rookie class generation + public views + scouting reports

Split out of the original draft.py.

This module owns:
- rookie bio/name/archetype generation
- true_ratings generation (private, not exposed to UI)
- college stats (public)
- combine metrics + workout report generation
- public prospect view helpers
- initialize_rookie_class_if_needed()
"""

from datetime import date
from typing import Any, Dict, List, Optional, Tuple
import random

from config import ROSTER_DF
from state import GAME_STATE

from draft_state import (
    _utc_iso,
    _safe_int,
    _clamp,
    _round1,
    _ensure_draft_state,
    OUTSIDE_SUB,
    INSIDE_SUB,
    PLAYMAKING_SUB,
    DEFENSE_SUB,
    REBOUNDING_SUB,
    ATHLETICISM_SUB,
    ALL_SUB_RATINGS,
    ALL_RATING_COLS,
    reserve_player_ids,
    validate_draft_integrity,
)

# ---------------------------------------------------------------------------
# 2) Rookie Bio / Name / Archetype
# ---------------------------------------------------------------------------

FIRST_NAMES = [
    "Jalen", "Marcus", "Tyrese", "Devin", "Brandon", "Caleb", "Darius", "Malik",
    "Cameron", "Jordan", "Anthony", "Isaiah", "Jaylen", "Noah", "Ethan", "Aiden",
    "Kai", "Liam", "Mason", "Julian", "Trey", "Zion", "Cade", "Jabari", "Keegan",
]
LAST_NAMES = [
    "Johnson", "Williams", "Brown", "Davis", "Miller", "Jones", "Anderson", "Moore",
    "Taylor", "Thomas", "Jackson", "White", "Harris", "Clark", "Lewis", "Walker",
    "Young", "King", "Wright", "Scott", "Green", "Baker", "Carter", "Collins",
]
COLLEGES = [
    "Duke", "Kentucky", "Kansas", "UCLA", "North Carolina", "Michigan State",
    "Gonzaga", "Arizona", "Baylor", "Villanova", "Texas", "USC", "Oregon",
    "Florida", "Tennessee", "Auburn", "Houston", "Virginia",
]

POSITIONS = ["PG", "SG", "SF", "PF", "C"]
ARCHETYPES = [
    "Volume Scorer",
    "Shooter",
    "Athletic Slasher",
    "Playmaker",
    "3&D Wing",
    "Defensive Anchor",
    "Rebounding Big",
    "Post Scorer",
    "Glue Guy",
    "Boom/Bust Project",
]


def _unique_name(rng: random.Random, used: set[str]) -> str:
    for _ in range(5000):
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if name not in used:
            used.add(name)
            return name
    # 극단적으로 이름이 소진되면 suffix 추가
    base = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
    i = 2
    while f"{base} {i}" in used:
        i += 1
    name = f"{base} {i}"
    used.add(name)
    return name


def _random_position(rng: random.Random) -> str:
    # 현실감: 가드/윙이 더 많고, 센터는 적게
    r = rng.random()
    if r < 0.22:
        return "PG"
    if r < 0.47:
        return "SG"
    if r < 0.72:
        return "SF"
    if r < 0.90:
        return "PF"
    return "C"


def _random_archetype_for_pos(rng: random.Random, pos: str) -> str:
    pool = list(ARCHETYPES)
    # 센터/빅에 가중
    if pos == "C":
        pool += ["Defensive Anchor", "Rebounding Big", "Post Scorer"] * 2
    if pos in {"PF", "C"}:
        pool += ["Rebounding Big", "Post Scorer"] * 2
    if pos in {"PG"}:
        pool += ["Playmaker", "Boom/Bust Project"] * 2
    if pos in {"SG", "SF"}:
        pool += ["Shooter", "3&D Wing", "Volume Scorer"] * 2
    return rng.choice(pool)


def _height_weight_from_pos(rng: random.Random, pos: str) -> Tuple[int, int]:
    """
    height_in: inches, weight_lbs: lbs
    매우 러프한 분포(컴바인 수치 생성의 기반)
    """
    if pos == "PG":
        h = rng.randint(71, 77)  # 5'11 ~ 6'5
        w = rng.randint(170, 205)
    elif pos == "SG":
        h = rng.randint(73, 79)
        w = rng.randint(185, 220)
    elif pos == "SF":
        h = rng.randint(76, 82)
        w = rng.randint(205, 240)
    elif pos == "PF":
        h = rng.randint(79, 84)
        w = rng.randint(220, 260)
    else:  # C
        h = rng.randint(81, 87)
        w = rng.randint(235, 285)
    return h, w


def _wingspan_from_height(rng: random.Random, height_in: int, pos: str) -> int:
    # 평균적으로 키보다 2~6인치 길게, 빅은 더 길게 나올 수 있음
    base = rng.randint(0, 8)
    if pos in {"PF", "C"}:
        base += rng.randint(0, 3)
    return height_in + base


def _standing_reach(height_in: int, wingspan_in: int) -> int:
    # 대략적인 근사. (NBA combine 평균대비 큰 오차가 날 수 있으나 게임용)
    # 키 + 팔 길이/2 + 8.5 정도
    sr = height_in + int((wingspan_in - height_in) / 2) + 9
    return sr


# ---------------------------------------------------------------------------
# 3) True Ratings 생성(내부 비공개)
# ---------------------------------------------------------------------------

def _tiered_target_ovr(rng: random.Random, idx: int, total: int) -> int:
    """
    90명 루키 클래스에서 '현실적인' 분포를 만들기 위한 타겟 OVR 생성.
    idx는 0..total-1
    """
    # 순위는 단순 idx로 잡고 분포를 섞는다.
    p = idx / max(1, total - 1)
    # 상위 5%: 75~82
    if p < 0.06:
        return rng.randint(75, 82)
    # 상위 20%: 70~78
    if p < 0.22:
        return rng.randint(70, 78)
    # 중간 55%: 62~72
    if p < 0.77:
        return rng.randint(62, 72)
    # 하위 23%: 55~66
    return rng.randint(55, 66)


def _compute_group_and_ovr(ratings: Dict[str, float]) -> Dict[str, float]:
    def avg(cols: List[str]) -> float:
        vals = [ratings.get(c, 0.0) for c in cols]
        return sum(vals) / max(1, len(vals))

    outside = avg(OUTSIDE_SUB)
    inside = avg(INSIDE_SUB)
    play = avg(PLAYMAKING_SUB)
    defense = avg(DEFENSE_SUB)
    reb = avg(REBOUNDING_SUB)
    ath = avg(ATHLETICISM_SUB)

    ratings["Outside Scoring"] = _round1(outside)
    ratings["Inside Scoring"] = _round1(inside)
    ratings["Playmaking"] = _round1(play)
    ratings["Defense"] = _round1(defense)
    ratings["Rebounding"] = _round1(reb)
    ratings["Athleticism"] = _round1(ath)

    ovr = (outside + inside + play + defense + reb + ath) / 6.0
    ratings["OVR"] = _round1(ovr)
    return ratings


def _archetype_bias(archetype: str) -> Dict[str, Tuple[float, float]]:
    """
    archetype별로 어떤 세부 능력치를 밀어주거나 깎는지.
    반환: col -> (mean_add, std_add)
    """
    b: Dict[str, Tuple[float, float]] = {}

    def add(cols: List[str], mean: float, std: float) -> None:
        for c in cols:
            b[c] = (mean, std)

    if archetype == "Volume Scorer":
        add(["Close Shot", "Mid-Range Shot", "Draw Foul", "Offensive Consistency"], +6, 4)
        add(["Shot IQ", "Pass IQ"], -3, 3)
    elif archetype == "Shooter":
        add(["Three-Point Shot", "Free Throw", "Shot IQ"], +8, 4)
        add(["Strength", "Interior Defense"], -3, 3)
    elif archetype == "Athletic Slasher":
        add(["Layup", "Driving Dunk", "Speed", "Vertical", "Hustle"], +7, 4)
        add(["Free Throw", "Shot IQ"], -2, 3)
    elif archetype == "Playmaker":
        add(["Pass Accuracy", "Ball Handle", "Pass Vision", "Pass IQ", "Speed with Ball"], +7, 4)
        add(["Strength", "Block"], -2, 3)
    elif archetype == "3&D Wing":
        add(["Three-Point Shot", "Perimeter Defense", "Help Defense IQ", "Defensive Consistency"], +6, 4)
        add(["Ball Handle"], -2, 3)
    elif archetype == "Defensive Anchor":
        add(["Interior Defense", "Block", "Help Defense IQ", "Defensive Consistency"], +8, 4)
        add(["Three-Point Shot", "Ball Handle"], -4, 3)
    elif archetype == "Rebounding Big":
        add(["Offensive Rebound", "Defensive Rebound", "Strength", "Hustle"], +7, 4)
        add(["Three-Point Shot", "Speed with Ball"], -4, 3)
    elif archetype == "Post Scorer":
        add(["Post Hook", "Post Fade", "Post Control", "Hands", "Draw Foul"], +7, 4)
        add(["Perimeter Defense", "Speed"], -3, 3)
    elif archetype == "Glue Guy":
        add(["Shot IQ", "Pass IQ", "Help Defense IQ", "Pass Perception", "Defensive Consistency"], +6, 3)
        add(["Offensive Consistency", "Ball Handle"], -2, 3)
    elif archetype == "Boom/Bust Project":
        # 운동능력은 높지만 IQ/일관성은 들쭉날쭉
        add(["Speed", "Vertical", "Strength"], +7, 6)
        add(["Shot IQ", "Offensive Consistency", "Defensive Consistency"], -5, 6)

    return b


def _generate_true_ratings(
    rng: random.Random,
    target_ovr: int,
    pos: str,
    archetype: str,
) -> Dict[str, float]:
    """
    1) 타겟 OVR 근처의 그룹 능력치를 만들고
    2) 세부 능력치로 분배
    3) archetype bias를 적용
    4) 최종 group/OVR 재계산
    """
    # 기본 베이스: target을 중심으로 그룹별 약간의 편차
    base = float(target_ovr)
    group_base = {
        "outside": base + rng.gauss(0, 4),
        "inside": base + rng.gauss(0, 4),
        "play": base + rng.gauss(0, 4),
        "defense": base + rng.gauss(0, 4),
        "reb": base + rng.gauss(0, 4),
        "ath": base + rng.gauss(0, 4),
    }

    # 포지션에 따른 구조적 편차(가드: 플레이/외곽, 빅: 인사이드/리바운드/수비)
    if pos == "PG":
        group_base["play"] += 4
        group_base["outside"] += 2
        group_base["reb"] -= 3
        group_base["inside"] -= 1
    elif pos == "SG":
        group_base["outside"] += 3
        group_base["play"] += 1
        group_base["reb"] -= 2
    elif pos == "SF":
        group_base["outside"] += 1
        group_base["defense"] += 1
    elif pos == "PF":
        group_base["inside"] += 2
        group_base["reb"] += 3
        group_base["defense"] += 1
        group_base["outside"] -= 1
    else:  # C
        group_base["inside"] += 3
        group_base["reb"] += 4
        group_base["defense"] += 2
        group_base["outside"] -= 4
        group_base["play"] -= 2

    # 세부 능력치 분배 (그룹 베이스 주변)
    ratings: Dict[str, float] = {}

    def fill(cols: List[str], mean: float, std: float) -> None:
        for c in cols:
            ratings[c] = _clamp(mean + rng.gauss(0, std))

    fill(OUTSIDE_SUB, group_base["outside"], 6)
    fill(INSIDE_SUB, group_base["inside"], 6)
    fill(PLAYMAKING_SUB, group_base["play"], 6)
    fill(DEFENSE_SUB, group_base["defense"], 6)
    fill(REBOUNDING_SUB, group_base["reb"], 6)
    fill(ATHLETICISM_SUB, group_base["ath"], 6)

    # archetype bias 적용
    bias = _archetype_bias(archetype)
    for col, (m, s) in bias.items():
        ratings[col] = _clamp(ratings.get(col, base) + rng.gauss(m, s))

    # 메타(내구/포텐셜) 생성
    durability = _clamp(70 + rng.gauss(0, 12), 40, 99)
    ratings["Overall Durability"] = _round1(durability)

    # Potential: 분포는 넓게, 타겟 OVR과 완전 결합되지 않게(상관 약하게)
    # - 상위권도 낮은 포텐셜이 있을 수 있고, 하위권도 높은 포텐셜이 있을 수 있음.
    pot_center = 72 + (target_ovr - 65) * 0.35 + rng.gauss(0, 14)
    if archetype == "Boom/Bust Project":
        pot_center += 8
    potential = _clamp(pot_center, 45, 99)
    ratings["Potential"] = _round1(potential)

    # group/ovr 계산 및 라운딩
    ratings = _compute_group_and_ovr(ratings)

    # OVR를 타겟에 약하게 맞추기(너무 벗어나면 미세 조정)
    # - 조정은 "전체 스케일"로 수행(현실감 유지)
    delta = target_ovr - ratings["OVR"]
    if abs(delta) > 5:
        scale = 0.5  # 너무 과하면 반만 반영
        for c in ALL_SUB_RATINGS:
            ratings[c] = _clamp(ratings[c] + delta * scale)
        ratings = _compute_group_and_ovr(ratings)

    # 모든 세부/그룹/메타 컬럼을 소수 1자리로 통일
    for c in ALL_RATING_COLS:
        if c in ratings:
            ratings[c] = _round1(ratings[c])

    return ratings


# ---------------------------------------------------------------------------
# 4) College Stats (공개 정보; "착각" 유도)
# ---------------------------------------------------------------------------

def _college_context(rng: random.Random) -> Dict[str, Any]:
    """
    낮은 수준의 리그에서 어떤 방식으로 스탯이 과장/왜곡되는지를 만드는 잠재 변수.
    - role: star / primary / secondary / role
    - pace: 경기 템포
    - scheme boosts: 특정 스킬 과대평가
    """
    r = rng.random()
    if r < 0.25:
        role = "Star"
        minutes = rng.randint(31, 36)
        usage = rng.uniform(0.28, 0.36)
    elif r < 0.55:
        role = "Primary"
        minutes = rng.randint(28, 34)
        usage = rng.uniform(0.24, 0.31)
    elif r < 0.82:
        role = "Secondary"
        minutes = rng.randint(22, 30)
        usage = rng.uniform(0.18, 0.26)
    else:
        role = "Role"
        minutes = rng.randint(14, 24)
        usage = rng.uniform(0.12, 0.20)

    pace = rng.uniform(65, 78)  # possessions-ish
    scheme = rng.choice([
        "Spread", "P&R Heavy", "Post", "Transition", "Zone-Press", "Motion"
    ])

    # "college fit"은 OVR과 완전히 독립적인 랜덤 변수로 크게 넣어서 착각을 유도
    fit = rng.gauss(0, 1.0)

    return {
        "role": role,
        "minutes": minutes,
        "usage": usage,
        "pace": pace,
        "scheme": scheme,
        "fit": fit,
    }


def _make_college_stats(
    rng: random.Random,
    pos: str,
    archetype: str,
    true_ratings: Dict[str, float],
    bio: Dict[str, Any],
) -> Dict[str, Any]:
    """
    true_ratings를 기반으로 하지만, 낮은 수준/전술/역할/운 변수로 인해
    NBA 변환성과 강하게 고정되지 않게 만든 "대학 스탯".
    """
    ctx = _college_context(rng)
    minutes = ctx["minutes"]
    usage = ctx["usage"]
    fit = ctx["fit"]

    # 핵심 능력치(참조) — 의도적으로 '무작위 가중치' + 'fit'로 왜곡
    outside = true_ratings.get("Outside Scoring", 60.0)
    inside = true_ratings.get("Inside Scoring", 60.0)
    play = true_ratings.get("Playmaking", 60.0)
    defense = true_ratings.get("Defense", 60.0)
    reb = true_ratings.get("Rebounding", 60.0)
    ath = true_ratings.get("Athleticism", 60.0)

    # "college bully" 요소: 체격/운동능력은 대학에서 더 잘 통한다(과대평가)
    height_in = _safe_int(bio.get("height_in"), 78)
    size_factor = (height_in - 77) / 6.0  # 대략 -1~+1

    # 득점: (outside + inside + ath)/3 기반이지만 fit/usage/role에 크게 좌우
    scoring_skill = (0.34 * outside + 0.34 * inside + 0.32 * ath)
    scheme = ctx["scheme"]
    if scheme == "Spread":
        scoring_skill += 0.20 * outside
    elif scheme == "Post":
        scoring_skill += 0.20 * inside + 3 * size_factor
    elif scheme == "Transition":
        scoring_skill += 0.20 * ath
    elif scheme == "P&R Heavy":
        scoring_skill += 0.10 * play + 0.10 * outside
    elif scheme == "Zone-Press":
        scoring_skill += 0.10 * ath + 0.10 * defense
    else:  # Motion
        scoring_skill += 0.10 * play + 0.10 * outside

    # 착각 유도: fit이 스탯을 크게 흔든다.
    scoring_skill += 8.0 * fit + rng.gauss(0, 6)

    # 효율(슈팅%)은 outside/inside와 상관 있지만, 대학이라 운/상대수준 영향이 큼
    fg = 0.40 + (inside - 60) * 0.003 + (ath - 60) * 0.001 + rng.gauss(0, 0.03)
    fg = max(0.32, min(0.68, fg))
    tp = 0.28 + (true_ratings.get("Three-Point Shot", 60) - 60) * 0.004 + rng.gauss(0, 0.04)
    tp = max(0.18, min(0.50, tp))
    ft = 0.62 + (true_ratings.get("Free Throw", 60) - 60) * 0.004 + rng.gauss(0, 0.03)
    ft = max(0.45, min(0.92, ft))

    # 분당 득점량: 사용률과 스킬 기반 (대학은 평균 득점이 높게 나오도록)
    pts_per_min = (scoring_skill / 100.0) * (0.35 + usage) * (0.85 + rng.uniform(-0.10, 0.10))
    # role이 낮으면 상한이 낮다
    if ctx["role"] in {"Role"}:
        pts_per_min *= 0.75
    pts = pts_per_min * minutes

    # 어시스트: play 기반이나, 포지션/role에 크게 의존
    ast = (0.015 * play + 0.004 * ath) * minutes + rng.gauss(0, 1.2)
    if pos == "PG":
        ast += 1.8
    elif pos == "SG":
        ast += 0.6
    elif pos in {"PF", "C"}:
        ast -= 0.6
    # 착각: "ball-dominant"면 어시스트도 뻥튀기
    ast += 6.0 * (usage - 0.22)

    # 리바운드: reb/size 기반, fit 영향 적게(그래도 운을 넣음)
    reb_base = (0.014 * reb + 0.0035 * ath + 0.004 * (height_in - 75)) * minutes
    # 가드면 리바운드 상한 낮게
    if pos in {"PG", "SG"}:
        reb_base *= 0.78
    reb_g = reb_base + rng.gauss(0, 1.4)

    # 스틸/블락: defense/ath 기반. 대학에서 프레스면 스틸 과대
    stl = (0.006 * defense + 0.004 * ath) * minutes + rng.gauss(0, 0.7)
    blk = (0.004 * defense + 0.004 * (height_in - 74)) * minutes + rng.gauss(0, 0.6)
    if scheme == "Zone-Press":
        stl += 0.8 + rng.random() * 0.6

    tov = max(0.6, (usage * 3.4) + rng.gauss(0, 0.7))  # usage가 높을수록 턴오버 증가

    games = rng.randint(28, 36)

    stats = {
        "G": games,
        "MIN": round(minutes, 1),
        "PTS": round(max(0.0, pts), 1),
        "REB": round(max(0.0, reb_g), 1),
        "AST": round(max(0.0, ast), 1),
        "STL": round(max(0.0, stl), 1),
        "BLK": round(max(0.0, blk), 1),
        "TOV": round(max(0.0, tov), 1),
        "FG%": round(fg, 3),
        "3P%": round(tp, 3),
        "FT%": round(ft, 3),
        # 공개/기획용: "겉보기 기대감" 정도로 쓸 수 있는 레벨(정답이 아님)
        "Role": ctx["role"],
        "Scheme": ctx["scheme"],
    }
    return stats


# ---------------------------------------------------------------------------
# 5) Combine Metrics (공개 정보; 요청 시 생성)
# ---------------------------------------------------------------------------

def _combine_metrics(
    rng: random.Random,
    pos: str,
    bio: Dict[str, Any],
    true_ratings: Dict[str, float],
) -> Dict[str, Any]:
    """
    Standing Reach / Height W/O Shoes / Wingspan / Verticals / Agility / Sprint
    반환값은 "좋을수록 큰 값"(vertical) 또는 "좋을수록 작은 값"(times) 혼재.
    """
    height_in = _safe_int(bio.get("height_in"))
    wingspan_in = _safe_int(bio.get("wingspan_in"))
    standing_reach_in = _safe_int(bio.get("standing_reach_in"))
    weight_lbs = _safe_int(bio.get("weight_lbs"))

    speed = true_ratings.get("Speed", 60.0)
    agility = true_ratings.get("Agility", 60.0)
    strength = true_ratings.get("Strength", 60.0)
    vertical = true_ratings.get("Vertical", 60.0)
    stamina = true_ratings.get("Stamina", 60.0)

    # Verticals (inches)
    # 키/몸무게가 크면 수치 불리하게
    size_penalty = (height_in - 77) * 0.35 + (weight_lbs - 220) * 0.03
    stand_vert = 18 + (vertical - 50) * 0.35 - size_penalty + rng.gauss(0, 1.6)
    max_vert = stand_vert + 8 + (speed - 50) * 0.10 + rng.gauss(0, 1.2)

    stand_vert = max(10.0, min(40.0, stand_vert))
    max_vert = max(18.0, min(48.0, max_vert))

    # Times (seconds; lower is better)
    # baseline from combine-ish ranges
    sprint_34 = 3.55 + (78 - speed) * 0.012 + (height_in - 77) * 0.006 + rng.gauss(0, 0.04)
    shuttle = 4.20 + (78 - agility) * 0.013 + (height_in - 77) * 0.004 + rng.gauss(0, 0.05)
    lane = 10.8 + (78 - agility) * 0.025 + (height_in - 77) * 0.012 + rng.gauss(0, 0.12)

    # strength가 너무 낮으면 가속이 떨어지는 느낌 반영
    sprint_34 += (70 - strength) * 0.0015

    sprint_34 = max(3.10, min(4.10, sprint_34))
    shuttle = max(3.60, min(4.80, shuttle))
    lane = max(9.8, min(12.6, lane))

    return {
        "Standing Reach": standing_reach_in,
        "Height W/O Shoes": height_in,
        "Wingspan": wingspan_in,
        "Standing Vertical Leap": round(stand_vert, 1),
        "Max Vertical Leap": round(max_vert, 1),
        "Shuttle Run": round(shuttle, 2),
        "Lane Agility": round(lane, 2),
        "Three Quarter Sprint": round(sprint_34, 2),
    }


# ---------------------------------------------------------------------------
# 6) Workout Report (공개 정보; 요청 시 생성)
# ---------------------------------------------------------------------------

def _workout_report(
    rng: random.Random,
    my_team_id: str,
    prospect_public: Dict[str, Any],
    true_ratings: Dict[str, float],
) -> Dict[str, Any]:
    """
    구단 연습경기/워크아웃을 '텍스트 리포트'로 제공.
    - true ratings를 직접 노출하지 않고
    - 관찰 기반(강점/약점/리스크/역할)로 작성 + 약간의 오차/노이즈
    """
    outside = true_ratings.get("Outside Scoring", 60.0)
    inside = true_ratings.get("Inside Scoring", 60.0)
    play = true_ratings.get("Playmaking", 60.0)
    defense = true_ratings.get("Defense", 60.0)
    reb = true_ratings.get("Rebounding", 60.0)
    ath = true_ratings.get("Athleticism", 60.0)
    pot = true_ratings.get("Potential", 70.0)

    # 관찰 노이즈: 스카우팅은 완벽하지 않다.
    noise = rng.gauss(0, 5)

    # "관찰 점수"를 만들되, 실제 수치 대신 구간/표현으로 내려준다.
    def band(x: float) -> str:
        x = x + noise
        if x >= 80:
            return "엘리트"
        if x >= 72:
            return "강점"
        if x >= 64:
            return "평균"
        if x >= 56:
            return "약점"
        return "큰 약점"

    strengths = []
    weaknesses = []

    # 강점/약점 토큰
    if outside + rng.gauss(0, 3) >= 72:
        strengths.append("외곽 슈팅/스페이싱 기여 가능")
    if inside + rng.gauss(0, 3) >= 72:
        strengths.append("림어택/피니시에서 경쟁력")
    if play + rng.gauss(0, 3) >= 72:
        strengths.append("볼 핸들링/패싱으로 2차 창출 가능")
    if defense + rng.gauss(0, 3) >= 72:
        strengths.append("수비 집중도와 포지셔닝이 인상적")
    if reb + rng.gauss(0, 3) >= 72:
        strengths.append("리바운드 세컨드찬스 창출 기대")
    if ath + rng.gauss(0, 3) >= 72:
        strengths.append("운동능력으로 트랜지션/수비 범위 확보")

    if outside + rng.gauss(0, 3) < 62:
        weaknesses.append("점프슛의 일관성이 부족해 보임")
    if inside + rng.gauss(0, 3) < 62:
        weaknesses.append("피니시/컨택 마무리에서 불안 요소")
    if play + rng.gauss(0, 3) < 62:
        weaknesses.append("드리블 압박에서 턴오버 위험")
    if defense + rng.gauss(0, 3) < 62:
        weaknesses.append("수비 상황 판단에서 기복")
    if reb + rng.gauss(0, 3) < 62:
        weaknesses.append("박스아웃/리바운드 기여도가 제한적")
    if ath + rng.gauss(0, 3) < 62:
        weaknesses.append("순간 가속/민첩성에서 한계")

    if not strengths:
        strengths.append("특정 한 가지보다는 다방면에서 무난한 인상")
    if not weaknesses:
        weaknesses.append("큰 약점은 보이지 않으나, 상위 레벨 적응은 관건")

    risk = "낮음"
    if pot >= 85 and (outside + inside + play + defense + reb + ath) / 6.0 < 68:
        risk = "높음(프로젝트 성격)"
    elif pot < 70 and ((outside + inside + play + defense + reb + ath) / 6.0) >= 72:
        risk = "중간(즉시전력이나 성장 제한 가능)"
    elif rng.random() < 0.12:
        risk = "중간(기복/역할 적응 변수)"

    # 종합 등급(정답 아님)
    score = (0.18 * outside + 0.18 * inside + 0.16 * play + 0.18 * defense + 0.12 * reb + 0.18 * ath) + rng.gauss(0, 4)
    if score >= 78:
        grade = "A-"
    elif score >= 74:
        grade = "B+"
    elif score >= 70:
        grade = "B"
    elif score >= 66:
        grade = "B-"
    elif score >= 62:
        grade = "C+"
    else:
        grade = "C"

    return {
        "team": my_team_id,
        "grade": grade,
        "risk": risk,
        "summary": {
            "Outside": band(outside),
            "Inside": band(inside),
            "Playmaking": band(play),
            "Defense": band(defense),
            "Rebounding": band(reb),
            "Athleticism": band(ath),
            "Potential": band(pot),
        },
        "strengths": strengths,
        "weaknesses": weaknesses,
        "generated_at": _utc_iso(),
    }


# ---------------------------------------------------------------------------
# 7) Public View Helpers (true_ratings 제외)
# ---------------------------------------------------------------------------

PUBLIC_PROSPECT_FIELDS = [
    "id", "name", "pos", "age", "college", "archetype",
    "college_stats",
    "combine", "workout_report",
    "drafted_by", "draft_pick",
]


def _public_prospect_view(p: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "id": p["id"],
        "name": p["name"],
        "pos": p["pos"],
        "age": p["age"],
        "college": p.get("college"),
        "archetype": p.get("archetype"),
        "college_stats": p.get("college_stats"),
        "drafted_by": p.get("drafted_by"),
        "draft_pick": p.get("draft_pick"),
    }
    # 요청에 의해 공개된 정보만 포함
    if p.get("reveals", {}).get("combine"):
        out["combine"] = p.get("combine")
    if p.get("reveals", {}).get("workout"):
        out["workout_report"] = p.get("workout_report")
    return out


def get_public_prospect_board() -> Dict[str, Any]:
    """
    전체 루키 보드를 '공개 가능한 정보만' 반환한다.
    (true ratings는 절대 포함하지 않음)
    """
    draft = _ensure_draft_state()
    board = []
    for pid in draft.get("board", []):
        p = draft["prospects"].get(str(pid))
        if not p:
            continue
        board.append(_public_prospect_view(p))
    return {
        "phase": draft.get("phase"),
        "season_id": draft.get("season_id"),
        "count": len(board),
        "prospects": board,
        "scouting": draft.get("scouting", {}),
        "lottery": {
            "ran_at": draft.get("lottery", {}).get("ran_at"),
            "top4": draft.get("lottery", {}).get("top4"),
            "draft_order_round1": draft.get("lottery", {}).get("draft_order_round1"),
        },
        "draft": {
            "current_pick_index": draft.get("draft", {}).get("current_pick_index"),
            "order": draft.get("draft", {}).get("order"),
            "picks": draft.get("draft", {}).get("picks"),
            "completed": draft.get("draft", {}).get("completed"),
        },
    }


def get_public_prospect(prospect_id: int) -> Dict[str, Any]:
    draft = _ensure_draft_state()
    p = draft["prospects"].get(str(prospect_id))
    if not p:
        raise KeyError(f"Prospect not found: {prospect_id}")
    return _public_prospect_view(p)


# ---------------------------------------------------------------------------
# 8) Rookie Class 생성/초기화
# ---------------------------------------------------------------------------

def _next_player_id() -> int:
    """(Deprecated) Use reserve_player_ids() instead.

    Kept for compatibility inside this file; returns one unique player id.
    """
    return reserve_player_ids(1)
       


def _season_id_from_date(d: Optional[str]) -> str:
    """
    리그 current_date(YYYY-MM-DD) 기반 시즌 id 문자열 생성.
    단순화: 10월 이전이면 전 시즌, 10월 이후면 해당 시즌 시작.
    """
    try:
        if d:
            y, m, _ = [int(x) for x in d.split("-")]
        else:
            today = date.today()
            y, m = today.year, today.month
    except Exception:
        today = date.today()
        y, m = today.year, today.month

    # NBA 시즌은 대략 10월 시작
    if m >= 10:
        return f"{y}-{str(y+1)[-2:]}"
    return f"{y-1}-{str(y)[-2:]}"


def initialize_rookie_class_if_needed(
    my_team_id: Optional[str] = None,
    seed: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    루키 클래스(90명)를 생성하고, 대학 스탯까지 포함해 드래프트 상태에 저장한다.
    - force=True면 기존 데이터가 있어도 다시 생성.
    - my_team_id는 스카우팅 요청 한도의 기준 팀(유저 팀)을 기록하는 용도.
    """
    draft = _ensure_draft_state()

    if not force and draft.get("phase") not in {None, "idle"} and draft.get("prospects"):
        # 이미 생성됨
        if my_team_id and not draft.get("scouting", {}).get("my_team_id"):
            draft["scouting"]["my_team_id"] = my_team_id
        return get_public_prospect_board()

    # 새 시즌 id / rng seed 세팅
    current_date = GAME_STATE.get("league", {}).get("current_date") or GAME_STATE.get("current_date")
    season_id = _season_id_from_date(current_date)
    draft["season_id"] = season_id
    draft["created_at"] = _utc_iso()
    draft["phase"] = "class_generated"

    rng_seed = seed if seed is not None else random.randint(1, 2_000_000_000)
    draft["rng_seed"] = int(rng_seed)
    rng = random.Random(draft["rng_seed"])

    # 유저 팀
    if my_team_id:
        draft["scouting"]["my_team_id"] = my_team_id

    # prospect 생성
    used_names: set[str] = set()
    prospects: Dict[str, Any] = {}
    board: List[int] = []

    base_pid = reserve_player_ids(90)  # roster와 id 충돌을 피하려면 별도 네임스페이스(또는 uuid)가 더 안전하지만,
                                 # 현재 프로젝트는 DF index 기반이어서 여기서도 정수 id를 사용
    for i in range(90):
        pid = base_pid + i
        pos = _random_position(rng)
        archetype = _random_archetype_for_pos(rng, pos)
        name = _unique_name(rng, used_names)
        age = rng.randint(19, 23)
        college = rng.choice(COLLEGES)

        height_in, weight_lbs = _height_weight_from_pos(rng, pos)
        wingspan_in = _wingspan_from_height(rng, height_in, pos)
        standing_reach_in = _standing_reach(height_in, wingspan_in)

        target_ovr = _tiered_target_ovr(rng, i, 90)
        true_r = _generate_true_ratings(rng, target_ovr, pos, archetype)

        bio = {
            "height_in": height_in,
            "weight_lbs": weight_lbs,
            "wingspan_in": wingspan_in,
            "standing_reach_in": standing_reach_in,
        }

        college_stats = _make_college_stats(rng, pos, archetype, true_r, bio)

        prospects[str(pid)] = {
            "id": pid,
            "name": name,
            "pos": pos,
            "age": age,
            "college": college,
            "archetype": archetype,
            "bio": bio,  # 내부에서만 사용(필요시 일부 공개)
            "true_ratings": true_r,  # ⚠️ 외부 비공개
            "college_stats": college_stats,  # 공개
            "combine": None,
            "workout_report": None,
            "reveals": {"combine": False, "workout": False},
            "drafted_by": None,
            "draft_pick": None,
        }
        board.append(pid)

    # 기본 보드 순서: 대학 스탯 기반으로 "겉보기" 정렬(착각 유도)
    def _public_score(pid: int) -> float:
        p = prospects[str(pid)]
        cs = p.get("college_stats") or {}
        # 대학 스탯의 pts/reb/ast + 효율 약간 반영. (정답 아님)
        return (
            1.0 * float(cs.get("PTS", 0))
            + 0.7 * float(cs.get("REB", 0))
            + 0.9 * float(cs.get("AST", 0))
            + 10.0 * float(cs.get("FG%", 0))
            + 6.0 * float(cs.get("3P%", 0))
        ) + rng.gauss(0, 2.0)

    board.sort(key=_public_score, reverse=True)

    draft["prospects"] = prospects
    draft["board"] = board

    # scouting reset
    draft["scouting"]["requested"] = []
    draft["scouting"]["combine_revealed"] = []
    draft["scouting"]["workout_revealed"] = []

    # lottery/draft reset
    draft["lottery"] = {
        "standings_worst_to_best": [],
        "draft_order_round1": [],
        "top4": [],
        "weights_1st_pick": [],
        "ran_at": None,
    }
    draft["draft"] = {
        "round": 1,
        "current_pick_index": 0,
        "order": [],
        "picks": [],
        "completed": False,
        "started_at": None,
        "completed_at": None,
    }

    validate_draft_integrity(stage="after_initialize_rookie_class", strict=True)

    return get_public_prospect_board()



