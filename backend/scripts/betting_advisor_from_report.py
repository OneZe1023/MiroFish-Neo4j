"""
Generate betting advice from a MiroFish report and an odds table.

Supports:
  - simple CSV/JSON odds: market,selection,line,odds
  - Sporttery CSV: pool_code,selection_code,goal_line,odds, ...

Optional LLM summary:
  --use-llm reads LLM_API_KEY / LLM_BASE_URL / LLM_MODEL_NAME from .env
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class OddsQuote:
    market: str
    selection: str
    odds: float
    line: Optional[float] = None
    bookmaker: str = ""


@dataclass
class Advice:
    market: str
    selection: str
    line: Optional[float]
    odds: float
    model_probability: float
    implied_probability: float
    edge: float
    fair_odds: float
    confidence: str
    recommendation: str
    stake_units: float
    reason: str


class ReportParseError(RuntimeError):
    pass


class BettingAdvisorFromReport:
    MIN_EDGE = 0.03
    STRONG_EDGE = 0.07
    MAX_STAKE_UNITS = 3.0

    SELECTION_ALIASES = {
        "home": {"home", "主胜", "home_win", "h", "1"},
        "draw": {"draw", "平局", "x"},
        "away": {"away", "客胜", "away_win", "a", "2"},
        "over": {"over", "大球", "over_2.5", "o"},
        "under": {"under", "小球", "under_2.5", "u"},
        "btts_yes": {"btts_yes", "yes", "双方进球_yes", "both_teams_score_yes"},
        "btts_no": {"btts_no", "no", "双方进球_no", "both_teams_score_no"},
        "home_or_draw": {"home_or_draw", "1x", "主胜或平", "home_draw"},
        "home_or_away": {"home_or_away", "12", "主胜或客胜", "home_away"},
        "draw_or_away": {"draw_or_away", "x2", "平或客胜", "draw_away"},
        "dnb_home": {"dnb_home", "home_dnb", "主队dnb"},
        "dnb_away": {"dnb_away", "away_dnb", "客队dnb"},
    }

    @classmethod
    def generate(cls, report_text: str, odds: List[OddsQuote]) -> Dict[str, Any]:
        model = cls.parse_report(report_text)
        advice = [cls.evaluate_quote(quote, model) for quote in odds]
        advice = [item for item in advice if item is not None]
        advice.sort(key=lambda item: item.edge, reverse=True)
        return {
            "match": {
                "home_team": model.get("home_team"),
                "away_team": model.get("away_team"),
            },
            "model_probabilities": model.get("probabilities", {}),
            "advice": [asdict(item) for item in advice],
            "top_value": asdict(advice[0]) if advice and advice[0].edge > 0 else None,
            "disclaimer": "仅基于报告模型概率与盘口赔率差值分析，不构成确定收益或保证性投注建议。",
        }

    @classmethod
    def parse_report(cls, text: str) -> Dict[str, Any]:
        home_team, away_team = cls._extract_teams(text)
        probabilities: Dict[str, float] = {}

        probabilities["home"] = cls._extract_percent_flexible(
            rf"-\s*{re.escape(home_team)}\s*主胜:\s*([0-9]+(?:\.[0-9]+)?)%",
            text,
            "主胜概率",
            [r"主胜(?:概率)?(?:为|约|:|：)?\s*([0-9]+(?:\.[0-9]+)?)%"],
        )
        probabilities["draw"] = cls._extract_percent_flexible(
            r"-\s*平局:\s*([0-9]+(?:\.[0-9]+)?)%",
            text,
            "平局概率",
            [r"平局(?:概率)?(?:为|约|:|：)?\s*([0-9]+(?:\.[0-9]+)?)%"],
        )
        probabilities["away"] = cls._extract_percent_flexible(
            rf"-\s*{re.escape(away_team)}\s*客胜:\s*([0-9]+(?:\.[0-9]+)?)%",
            text,
            "客胜概率",
            [r"客胜(?:概率)?(?:为|约|:|：)?\s*([0-9]+(?:\.[0-9]+)?)%"],
        )

        over_under = re.search(
            r"Over\s*2\.5\s*概率为\s*([0-9]+(?:\.[0-9]+)?)%.*?"
            r"Under\s*2\.5\s*概率为\s*([0-9]+(?:\.[0-9]+)?)%",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if over_under:
            probabilities["over_2.5"] = float(over_under.group(1)) / 100.0
            probabilities["under_2.5"] = float(over_under.group(2)) / 100.0

        btts = re.search(
            r"BTTS\s*Yes\s*概率为\s*([0-9]+(?:\.[0-9]+)?)%.*?"
            r"BTTS\s*No\s*概率为\s*([0-9]+(?:\.[0-9]+)?)%",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if btts:
            probabilities["btts_yes"] = float(btts.group(1)) / 100.0
            probabilities["btts_no"] = float(btts.group(2)) / 100.0

        dc = re.search(
            r"Double Chance：主胜或平\s*([0-9]+(?:\.[0-9]+)?)%，"
            r"主胜或客胜\s*([0-9]+(?:\.[0-9]+)?)%，"
            r"平或客胜\s*([0-9]+(?:\.[0-9]+)?)%。"
            r"DNB 市场中，主队不败结算概率为\s*([0-9]+(?:\.[0-9]+)?)%，"
            r"客队不败结算概率为\s*([0-9]+(?:\.[0-9]+)?)%",
            text,
            re.DOTALL,
        )
        if dc:
            probabilities["home_or_draw"] = float(dc.group(1)) / 100.0
            probabilities["home_or_away"] = float(dc.group(2)) / 100.0
            probabilities["draw_or_away"] = float(dc.group(3)) / 100.0
            probabilities["dnb_home"] = float(dc.group(4)) / 100.0
            probabilities["dnb_away"] = float(dc.group(5)) / 100.0

        score_matrix = cls._extract_score_matrix(text)
        if score_matrix:
            cls._add_score_derived_probabilities(probabilities, score_matrix)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "probabilities": probabilities,
            "score_matrix": score_matrix,
        }

    @classmethod
    def evaluate_quote(cls, quote: OddsQuote, model: Dict[str, Any]) -> Optional[Advice]:
        probability_key = cls._probability_key(quote)
        model_prob = model["probabilities"].get(probability_key)
        if model_prob is None:
            return None

        implied = cls._implied_probability(quote.odds)
        if implied <= 0:
            return None

        edge = model_prob - implied
        fair_odds = cls._fair_odds(model_prob)
        confidence = cls._confidence(edge)
        recommendation = "value_candidate" if edge >= cls.MIN_EDGE else "no_bet"
        stake = cls._stake_units(edge, quote.odds) if recommendation == "value_candidate" else 0.0
        reason = cls._reason(model_prob, implied, edge, quote.odds, recommendation)

        return Advice(
            market=quote.market,
            selection=quote.selection,
            line=quote.line,
            odds=round(quote.odds, 4),
            model_probability=round(model_prob, 4),
            implied_probability=round(implied, 4),
            edge=round(edge, 4),
            fair_odds=fair_odds,
            confidence=confidence,
            recommendation=recommendation,
            stake_units=stake,
            reason=reason,
        )

    @classmethod
    def _probability_key(cls, quote: OddsQuote) -> str:
        market = cls._normalize(quote.market)
        selection = cls._normalize(quote.selection)

        if market in {"1x2", "match_winner", "胜平负"}:
            return cls._canonical_selection(selection)
        if market in {"hhad", "handicap_1x2", "让球胜平负"}:
            side = cls._canonical_selection(selection)
            if quote.line is not None:
                return f"hhad_{quote.line:g}_{side}"
            return f"hhad_{side}"
        if market in {"ttg", "total_goals", "总进球"}:
            side = cls._canonical_selection(selection)
            if side.startswith("s"):
                return f"ttg_{side}"
            return side
        if market in {"over_under", "totals", "大小球"}:
            side = cls._canonical_selection(selection)
            line = quote.line if quote.line is not None else 2.5
            return f"{side}_{line:g}"
        if market in {"btts", "双方进球"}:
            side = cls._canonical_selection(selection)
            return f"btts_{side}" if side in {"yes", "no"} else side
        if market in {"double_chance", "双重机会"}:
            return cls._canonical_selection(selection)
        if market in {"dnb", "draw_no_bet", "平局退款"}:
            side = cls._canonical_selection(selection)
            return f"dnb_{side}" if side in {"home", "away"} else side
        return cls._canonical_selection(selection)

    @classmethod
    def _canonical_selection(cls, selection: str) -> str:
        normalized = cls._normalize(selection)
        for canonical, aliases in cls.SELECTION_ALIASES.items():
            if normalized in {cls._normalize(alias) for alias in aliases}:
                return canonical
        return normalized

    @classmethod
    def _extract_teams(cls, text: str) -> tuple[str, str]:
        home_match = re.search(r"-\s*主队:\s*(.+)", text, re.MULTILINE)
        away_match = re.search(r"-\s*客队:\s*(.+)", text, re.MULTILINE)
        if home_match and away_match:
            return cls._clean_team_name(home_match.group(1)), cls._clean_team_name(away_match.group(1))

        title_match = re.search(r"^#\s*(.+?)\s+vs\s+(.+?)(?:预测|报告|$)", text, re.IGNORECASE | re.MULTILINE)
        if title_match:
            return cls._clean_team_name(title_match.group(1)), cls._clean_team_name(title_match.group(2))

        raise ReportParseError("无法从报告中解析主队/客队；请使用 --home 和 --away 指定")

    @staticmethod
    def _clean_team_name(value: str) -> str:
        cleaned = str(value or "").strip()
        for phrase in ("在主场", "的比赛中被极度看好", "作为主队", "作为客队"):
            cleaned = cleaned.replace(phrase, "")
        return cleaned.strip(" ：:，,。")

    @staticmethod
    def _normalize(value: Any) -> str:
        return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _implied_probability(odds: float) -> float:
        return 1.0 / odds if odds and odds > 1.0 else 0.0

    @staticmethod
    def _fair_odds(probability: float) -> float:
        return round(1.0 / probability, 3) if probability > 0 else 999.0

    @classmethod
    def _confidence(cls, edge: float) -> str:
        if edge >= cls.STRONG_EDGE:
            return "strong"
        if edge >= cls.MIN_EDGE:
            return "moderate"
        if edge > 0:
            return "thin"
        return "none"

    @classmethod
    def _stake_units(cls, edge: float, odds: float) -> float:
        b = odds - 1.0
        if b <= 0:
            return 0.0
        p = edge + 1.0 / odds
        q = 1.0 - p
        kelly = max(0.0, (b * p - q) / b)
        return round(min(cls.MAX_STAKE_UNITS, kelly * 5.0), 2)

    @staticmethod
    def _reason(model_prob: float, implied: float, edge: float, odds: float, recommendation: str) -> str:
        if recommendation == "value_candidate":
            return (
                f"模型概率 {model_prob:.1%} 高于赔率 {odds:.2f} 的隐含概率 "
                f"{implied:.1%}，正向差值为 {edge:.1%}。"
            )
        return (
            f"模型概率 {model_prob:.1%} 未显著高于赔率 {odds:.2f} 的隐含概率 "
            f"{implied:.1%}，暂不构成下注候选。"
        )

    @staticmethod
    def _match_required(pattern: str, text: str, label: str) -> str:
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ReportParseError(f"无法从报告中解析 {label}")
        return match.group(1).strip()

    @classmethod
    def _extract_percent_flexible(
        cls,
        pattern: str,
        text: str,
        label: str,
        fallbacks: Optional[List[str]] = None,
    ) -> float:
        try:
            return cls._extract_percent(pattern, text, label)
        except ReportParseError:
            for fallback in fallbacks or []:
                match = re.search(fallback, text, re.MULTILINE | re.DOTALL)
                if match:
                    return float(match.group(1)) / 100.0
            raise

    @staticmethod
    def _extract_percent(pattern: str, text: str, label: str) -> float:
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            raise ReportParseError(f"无法从报告中解析 {label}")
        return float(match.group(1)) / 100.0

    @staticmethod
    def _extract_score_matrix(text: str) -> Dict[tuple[int, int], float]:
        rows: Dict[tuple[int, int], float] = {}
        table_match = re.search(
            r"\|\s*主队\\客队\s*\|(?P<header>.+?)\|\s*\n"
            r"\|[-|\s:]+\|\s*\n"
            r"(?P<body>(?:\|\s*\d+\s*\|.+?\|\s*\n)+)",
            text,
            re.MULTILINE,
        )
        if not table_match:
            return rows

        away_goals = [int(item.strip()) for item in table_match.group("header").split("|") if item.strip().isdigit()]
        for line in table_match.group("body").splitlines():
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            if len(cells) < 2 or not cells[0].isdigit():
                continue
            home_goals = int(cells[0])
            for away_goals_value, raw in zip(away_goals, cells[1:]):
                pct_match = re.search(r"([0-9]+(?:\.[0-9]+)?)%", raw)
                if pct_match:
                    rows[(home_goals, away_goals_value)] = float(pct_match.group(1)) / 100.0
        return rows

    @staticmethod
    def _add_score_derived_probabilities(
        probabilities: Dict[str, float],
        score_matrix: Dict[tuple[int, int], float],
    ) -> None:
        totals = {str(i): 0.0 for i in range(7)}
        totals["7"] = 0.0
        for (home_goals, away_goals), prob in score_matrix.items():
            total_goals = home_goals + away_goals
            key = str(total_goals) if total_goals <= 6 else "7"
            totals[key] += prob
        for key, prob in totals.items():
            probabilities[f"ttg_s{key}"] = prob

    @classmethod
    def build_llm_summary(cls, result: Dict[str, Any], report_text: str, max_items: int = 12) -> str:
        if OpenAI is None:
            raise RuntimeError("openai package is not installed; cannot use --use-llm")

        env_path = Path(__file__).resolve().parents[2] / ".env"
        load_env_file(env_path)
        api_key = os.environ.get("LLM_API_KEY", "")
        base_url = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
        model = os.environ.get("LLM_MODEL_NAME", "gpt-4o-mini")
        if not api_key:
            raise RuntimeError("LLM_API_KEY 未配置，无法调用大模型")

        compact = {
            "match": result["match"],
            "model_probabilities": result["model_probabilities"],
            "top_value": result["top_value"],
            "advice": result["advice"][:max_items],
            "disclaimer": result["disclaimer"],
        }
        prompt = (
            "你是足球交易风控分析助手。请基于给定的量化结果生成中文下注建议，"
            "不要修改概率、赔率、edge 或 stake 数值，不要承诺收益。"
            "输出包含：核心结论、弱候选但不下注、可下注候选、回避项、资金分配、风险提示。"
            "如果 advice 中存在 edge > 0 但 recommendation=no_bet 的项目，必须归入“弱候选但不下注”，"
            "不要称为“无候选”；只有 recommendation=value_candidate 的项目才可写入“可下注候选”。\n\n"
            f"量化结果JSON:\n{json.dumps(compact, ensure_ascii=False, indent=2)}\n\n"
            f"报告摘录:\n{report_text[:6000]}"
        )
        client = OpenAI(api_key=api_key, base_url=base_url)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你只基于用户提供的数据做投注辅助分析，保持谨慎、可审计。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1800,
        )
        content = response.choices[0].message.content or ""
        content = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
        return content


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_odds(
    path: Path,
    match_id: str = "",
    home: str = "",
    away: str = "",
    markets: Optional[set[str]] = None,
) -> List[OddsQuote]:
    if not path.exists():
        raise FileNotFoundError(f"盘口赔率文件不存在: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = data.get("odds", data) if isinstance(data, dict) else data
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))

    quotes: List[OddsQuote] = []
    for row in rows:
        if not row_matches(row, match_id=match_id, home=home, away=away):
            continue
        quote = row_to_quote(row)
        if quote is None:
            continue
        if markets and BettingAdvisorFromReport._normalize(quote.market) not in markets:
            continue
        quotes.append(quote)
    return quotes


def row_to_quote(row: Dict[str, Any]) -> Optional[OddsQuote]:
    if "pool_code" in row and "selection_code" in row:
        odds = row.get("odds")
        if odds in (None, ""):
            return None
        market, selection, line = sporttery_market_selection(
            str(row.get("pool_code", "")).strip().lower(),
            str(row.get("selection_code", "")).strip().lower(),
            row.get("goal_line"),
        )
        if not market:
            return None
        return OddsQuote(
            market=market,
            selection=selection,
            odds=float(odds),
            line=line,
            bookmaker="sporttery",
        )

    odds = row.get("odds") or row.get("decimal_odds")
    if odds in (None, ""):
        return None
    line = row.get("line")
    return OddsQuote(
        market=str(row.get("market", "")).strip(),
        selection=str(row.get("selection", "")).strip(),
        odds=float(odds),
        line=float(line) if line not in (None, "") else None,
        bookmaker=str(row.get("bookmaker", "")).strip(),
    )


def sporttery_market_selection(pool: str, selection_code: str, raw_goal_line: Any) -> tuple[str, str, Optional[float]]:
    if pool == "had":
        return "1x2", {"h": "home", "d": "draw", "a": "away"}.get(selection_code, selection_code), None
    if pool == "hhad":
        return "hhad", {"h": "home", "d": "draw", "a": "away"}.get(selection_code, selection_code), parse_goal_line(raw_goal_line)
    if pool == "ttg":
        return "ttg", selection_code, None
    if pool == "crs":
        return "crs", selection_code, None
    return "", "", None


def parse_goal_line(raw_goal_line: Any) -> Optional[float]:
    if raw_goal_line in (None, ""):
        return None
    try:
        return float(str(raw_goal_line).replace("+", ""))
    except ValueError:
        return None


def row_matches(row: Dict[str, Any], match_id: str = "", home: str = "", away: str = "") -> bool:
    if match_id:
        return str(row.get("match_id", "")).strip() == str(match_id).strip()
    if home and not fuzzy_contains(row.get("home_team_name", "") or row.get("home_team_abbr", ""), home):
        return False
    if away and not fuzzy_contains(row.get("away_team_name", "") or row.get("away_team_abbr", ""), away):
        return False
    return True


def fuzzy_contains(candidate: Any, expected: str) -> bool:
    left = normalize_team_text(candidate)
    right = normalize_team_text(expected)
    return bool(left and right and (left in right or right in left))


def normalize_team_text(value: Any) -> str:
    text = str(value or "").lower()
    for old in ("民主共和国", "共和国", "的比赛中被极度看好", "在主场", "(", ")", "（", "）", " "):
        text = text.replace(old, "")
    return text.strip()


def render_markdown(result: Dict[str, Any]) -> str:
    match = result["match"]
    lines = [
        f"# Betting Advice: {match.get('home_team')} vs {match.get('away_team')}",
        "",
        "| Market | Selection | Line | Odds | Model | Implied | Edge | Confidence | Recommendation | Stake |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---:|",
    ]
    for item in result["advice"]:
        line = "" if item["line"] is None else item["line"]
        lines.append(
            f"| {item['market']} | {item['selection']} | {line} | {item['odds']:.2f} | "
            f"{item['model_probability']:.1%} | {item['implied_probability']:.1%} | "
            f"{item['edge']:.1%} | {item['confidence']} | {item['recommendation']} | {item['stake_units']} |"
        )
    lines.extend(["", "## Notes", ""])
    for item in result["advice"]:
        lines.append(f"- **{item['market']} {item['selection']}**: {item['reason']}")
    lines.extend(["", f"> {result['disclaimer']}"])
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate betting advice from report markdown and odds.")
    parser.add_argument("--report", required=True, help="Path to full_report.md")
    parser.add_argument("--odds", required=True, help="Path to odds CSV or JSON")
    parser.add_argument("--output-json", default="", help="Output JSON path")
    parser.add_argument("--output-md", default="", help="Output Markdown path")
    parser.add_argument("--match-id", default="", help="Sporttery match_id to select from the odds CSV")
    parser.add_argument("--home", default="", help="Home team name to select from Sporttery CSV")
    parser.add_argument("--away", default="", help="Away team name to select from Sporttery CSV")
    parser.add_argument(
        "--markets",
        default="1x2,hhad,ttg",
        help="Comma-separated normalized markets to evaluate. Defaults to 1x2,hhad,ttg.",
    )
    parser.add_argument("--use-llm", action="store_true", help="Use .env LLM/MiniMax config to write a narrative summary")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report_path = Path(args.report)
    odds_path = Path(args.odds)
    report_text = report_path.read_text(encoding="utf-8")

    parsed = BettingAdvisorFromReport.parse_report(report_text)
    home = args.home or parsed.get("home_team", "")
    away = args.away or parsed.get("away_team", "")
    markets = {BettingAdvisorFromReport._normalize(item) for item in args.markets.split(",") if item.strip()}

    odds = load_odds(odds_path, match_id=args.match_id, home=home, away=away, markets=markets)
    if not odds:
        raise RuntimeError("没有从赔率 CSV 中匹配到可评估赔率；请尝试使用 --match-id 或 --home/--away 指定比赛")

    result = BettingAdvisorFromReport.generate(report_text, odds)
    output_json = Path(args.output_json) if args.output_json else report_path.with_name("betting_advice.json")
    output_md = Path(args.output_md) if args.output_md else report_path.with_name("betting_advice.md")

    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_markdown(result)
    if args.use_llm:
        markdown += "\n\n## LLM Trading Desk Summary\n\n"
        markdown += BettingAdvisorFromReport.build_llm_summary(result, report_text)
        markdown += "\n"
    output_md.write_text(markdown, encoding="utf-8")

    print(f"JSON: {output_json}")
    print(f"Markdown: {output_md}")
    if result["top_value"]:
        top = result["top_value"]
        print(f"Top value: {top['market']} {top['selection']} edge={top['edge']:.1%}")
    else:
        print("Top value: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
