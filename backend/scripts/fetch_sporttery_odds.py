"""
Fetch football odds from Sporttery's match calculator API.

Outputs two CSV files by default:
  - sporttery_odds.csv: HAD, HHAD, TTG, HAFU markets
  - sporttery_score_odds.csv: CRS correct-score market

The CSV columns intentionally stay close to the Sporttery API field names while
adding a few readable labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://webapi.sporttery.cn/gateway/uniform/football/getMatchCalculatorV1.qry"
REFERER = "https://m.sporttery.cn/mjc/jsq/zqspf/"
DEFAULT_POOLS = ("had", "hhad", "ttg", "hafu", "crs")
MAIN_POOLS = {"had", "hhad", "ttg", "hafu"}
SCORE_POOL = "crs"

HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://m.sporttery.cn",
    "Referer": REFERER,
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
        "Mobile/15E148 Safari/604.1"
    ),
}

POOL_LABELS = {
    "had": "胜平负",
    "hhad": "让球胜平负",
    "ttg": "总进球",
    "hafu": "半全场",
    "crs": "比分",
}

SELECTION_LABELS = {
    "had": {"h": "主胜", "d": "平", "a": "客胜"},
    "hhad": {"h": "让胜", "d": "让平", "a": "让负"},
    "ttg": {
        "s0": "0球",
        "s1": "1球",
        "s2": "2球",
        "s3": "3球",
        "s4": "4球",
        "s5": "5球",
        "s6": "6球",
        "s7": "7+球",
    },
    "hafu": {
        "hh": "胜胜",
        "hd": "胜平",
        "ha": "胜负",
        "dh": "平胜",
        "dd": "平平",
        "da": "平负",
        "ah": "负胜",
        "ad": "负平",
        "aa": "负负",
    },
    "crs": {},
}

SELECTION_ORDER = {
    "had": ("h", "d", "a"),
    "hhad": ("h", "d", "a"),
    "ttg": ("s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7"),
    "hafu": ("hh", "hd", "ha", "dh", "dd", "da", "ah", "ad", "aa"),
}

CRS_SCORE_CODES = [
    "s00s00",
    "s00s01",
    "s00s02",
    "s00s03",
    "s00s04",
    "s00s05",
    "s01s00",
    "s01s01",
    "s01s02",
    "s01s03",
    "s01s04",
    "s01s05",
    "s02s00",
    "s02s01",
    "s02s02",
    "s02s03",
    "s02s04",
    "s02s05",
    "s03s00",
    "s03s01",
    "s03s02",
    "s03s03",
    "s04s00",
    "s04s01",
    "s04s02",
    "s05s00",
    "s05s01",
    "s05s02",
    "s1sh",
    "s1sd",
    "s1sa",
]

CRS_SPECIAL_LABELS = {
    "s1sh": "胜其他",
    "s1sd": "平其他",
    "s1sa": "负其他",
}

CSV_COLUMNS = [
    "match_id",
    "match_num",
    "match_num_date",
    "business_date",
    "match_date",
    "match_time",
    "league_id",
    "league_name",
    "league_abbr",
    "home_team_id",
    "home_team_name",
    "home_team_abbr",
    "away_team_id",
    "away_team_name",
    "away_team_abbr",
    "pool_code",
    "pool_label",
    "pool_status",
    "single",
    "all_up",
    "cbt_value",
    "int_value",
    "vbt_value",
    "goal_line",
    "selection_code",
    "selection_label",
    "odds",
    "trend",
    "last_update_time",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch Sporttery football odds to CSV.")
    parser.add_argument("--out", default="sporttery_odds.csv", help="Main-market CSV path")
    parser.add_argument(
        "--score-out",
        default="sporttery_score_odds.csv",
        help="Correct-score CSV path; pass empty string to skip",
    )
    parser.add_argument(
        "--pools",
        default=",".join(DEFAULT_POOLS),
        help="Comma-separated Sporttery pool codes, e.g. had,hhad,ttg,hafu,crs",
    )
    parser.add_argument(
        "--league",
        help="Only export matches whose league name or abbreviation contains this text, e.g. 世界杯",
    )
    parser.add_argument("--raw-json", help="Optional path to save the raw API JSON")
    args = parser.parse_args()

    pools = [pool.strip().lower() for pool in args.pools.split(",") if pool.strip()]
    payload = fetch_payload(pools)

    if args.raw_json:
        raw_path = Path(args.raw_json)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    main_rows, score_rows = flatten_payload(payload, league_filter=args.league)
    write_csv(Path(args.out), main_rows)
    if args.score_out:
        write_csv(Path(args.score_out), score_rows)

    print(f"Wrote {len(main_rows)} rows to {args.out}")
    if args.score_out:
        print(f"Wrote {len(score_rows)} score rows to {args.score_out}")
    return 0


def fetch_payload(pools: Iterable[str]) -> Dict[str, Any]:
    query = urlencode({"channel": "c", "poolCode": ",".join(pools)})
    request = Request(f"{API_URL}?{query}", headers=HEADERS, method="GET")
    try:
        with urlopen(request, timeout=30) as response:
            text = response.read().decode("utf-8")
    except HTTPError as exc:
        raise RuntimeError(
            f"Sporttery API returned HTTP {exc.code}. "
            "If this changes back to 403, run from a real browser/MCP context."
        ) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach Sporttery API: {exc.reason}") from exc

    payload = json.loads(text)
    if str(payload.get("errorCode")) != "0":
        raise RuntimeError(f"Sporttery API error: {payload.get('errorMessage') or payload}")
    return payload


def flatten_payload(
    payload: Dict[str, Any],
    league_filter: Optional[str] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    value = payload.get("value") or {}
    last_update_time = value.get("lastUpdateTime", "")
    main_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []

    for group in value.get("matchInfoList") or []:
        for match in group.get("subMatchList") or []:
            if league_filter and not is_league_match(match, league_filter):
                continue
            pool_meta = {
                str(item.get("poolCode", "")).lower(): item for item in match.get("poolList") or []
            }
            base = match_base(match, last_update_time)

            for pool_code in MAIN_POOLS:
                odds_obj = match.get(pool_code)
                if not isinstance(odds_obj, dict) or not odds_obj:
                    continue
                meta = pool_meta.get(pool_code, {})
                for row in flatten_main_pool(base, pool_code, odds_obj, meta):
                    main_rows.append(row)

            crs = match.get(SCORE_POOL)
            if isinstance(crs, dict) and crs:
                meta = pool_meta.get(SCORE_POOL, {})
                for row in flatten_score_pool(base, crs, meta):
                    score_rows.append(row)

    return main_rows, score_rows


def is_league_match(match: Dict[str, Any], league_filter: str) -> bool:
    needle = league_filter.strip().lower()
    if not needle:
        return True
    haystack = " ".join(
        str(match.get(key, ""))
        for key in ("leagueAllName", "leagueAbbName", "leagueName", "leagueAbbEnName")
    ).lower()
    return needle in haystack


def match_base(match: Dict[str, Any], last_update_time: str) -> Dict[str, Any]:
    return {
        "match_id": match.get("matchId", ""),
        "match_num": match.get("matchNumStr", ""),
        "match_num_date": match.get("matchNumDate", ""),
        "business_date": match.get("businessDate", ""),
        "match_date": match.get("matchDate", ""),
        "match_time": match.get("matchTime", ""),
        "league_id": match.get("leagueId", ""),
        "league_name": match.get("leagueAllName", ""),
        "league_abbr": match.get("leagueAbbName", ""),
        "home_team_id": match.get("homeTeamId", ""),
        "home_team_name": match.get("homeTeamAllName", ""),
        "home_team_abbr": match.get("homeTeamAbbName", ""),
        "away_team_id": match.get("awayTeamId", ""),
        "away_team_name": match.get("awayTeamAllName", ""),
        "away_team_abbr": match.get("awayTeamAbbName", ""),
        "last_update_time": last_update_time,
    }


def flatten_main_pool(
    base: Dict[str, Any],
    pool_code: str,
    odds_obj: Dict[str, Any],
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for selection_code in SELECTION_ORDER[pool_code]:
        odds = odds_obj.get(selection_code)
        if odds in (None, ""):
            continue
        rows.append(
            make_row(
                base=base,
                pool_code=pool_code,
                selection_code=selection_code,
                selection_label=SELECTION_LABELS[pool_code][selection_code],
                odds=odds,
                trend=odds_obj.get(f"{selection_code}f", ""),
                goal_line=odds_obj.get("goalLine", ""),
                meta=meta,
            )
        )
    return rows


def flatten_score_pool(
    base: Dict[str, Any],
    odds_obj: Dict[str, Any],
    meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    for selection_code in CRS_SCORE_CODES:
        odds = odds_obj.get(selection_code)
        if odds in (None, ""):
            continue
        rows.append(
            make_row(
                base=base,
                pool_code=SCORE_POOL,
                selection_code=selection_code,
                selection_label=crs_label(selection_code),
                odds=odds,
                trend=odds_obj.get(f"{selection_code}f", ""),
                goal_line=odds_obj.get("goalLine", ""),
                meta=meta,
            )
        )
    return rows


def make_row(
    base: Dict[str, Any],
    pool_code: str,
    selection_code: str,
    selection_label: str,
    odds: Any,
    trend: Any,
    goal_line: Any,
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "pool_code": pool_code,
            "pool_label": POOL_LABELS.get(pool_code, pool_code),
            "pool_status": meta.get("poolStatus", ""),
            "single": meta.get("single", ""),
            "all_up": meta.get("allUp", ""),
            "cbt_value": meta.get("cbtValue", ""),
            "int_value": meta.get("intValue", ""),
            "vbt_value": meta.get("vbtValue", ""),
            "goal_line": goal_line,
            "selection_code": selection_code,
            "selection_label": selection_label,
            "odds": odds,
            "trend": trend,
        }
    )
    return row


def crs_label(code: str) -> str:
    if code in CRS_SPECIAL_LABELS:
        return CRS_SPECIAL_LABELS[code]
    if len(code) == 6 and code.startswith("s") and code[3] == "s":
        return f"{int(code[1:3])}:{int(code[4:6])}"
    return code


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    if sys.platform == "win32":
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
