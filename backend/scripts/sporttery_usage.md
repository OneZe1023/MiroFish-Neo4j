# Sporttery Script Commands

## 1) 抓取世界杯赔率

```powershell
cd E:\project\MiroFish\backend
uv run python .\scripts\fetch_sporttery_odds.py --league 世界杯 --out .\uploads\sporttery_worldcup_odds.csv --score-out .\uploads\sporttery_worldcup_score_odds.csv --raw-json .\uploads\sporttery_raw.json
```

## 2) 生成投注建议

```powershell
cd E:\project\MiroFish\backend
uv run python .\scripts\betting_advisor_from_report.py --report .\uploads\reports\report_4f78020d9c15\full_report.md --odds .\uploads\sporttery_worldcup_odds.csv --match-id 2040182 --output-json .\uploads\reports\report_4f78020d9c15\sporttery_betting_advice.json --output-md .\uploads\reports\report_4f78020d9c15\sporttery_betting_advice.md
```

## 3) 生成投注建议并调用大模型摘要

```powershell
cd E:\project\MiroFish\backend
uv run python .\scripts\betting_advisor_from_report.py --report .\uploads\reports\report_4f78020d9c15\full_report.md --odds .\uploads\sporttery_worldcup_odds.csv --match-id 2040182 --output-json .\uploads\reports\report_4f78020d9c15\sporttery_betting_advice.json --output-md .\uploads\reports\report_4f78020d9c15\sporttery_betting_advice.md --use-llm
```

## 4) 仅按球队名匹配

```powershell
cd E:\project\MiroFish\backend
uv run python .\scripts\betting_advisor_from_report.py --report .\uploads\reports\report_4f78020d9c15\full_report.md --odds .\uploads\sporttery_worldcup_odds.csv --home 葡萄牙 --away 刚果 --use-llm
```

