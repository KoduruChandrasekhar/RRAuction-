#!/usr/bin/env bash
# Rebuilds every output from the raw data pack. Re-run after editing analysis/availability_config.json,
# analysis/prices.json or analysis/salaries_2026.json (e.g. when official retention lists land).
# Usage: bash run_all.sh [--skip-models]     (--skip-models reuses trained models; ~6 min instead of ~15)
set -e
cd "$(dirname "$0")"
DATA=data_pack; W=outputs/work; S=outputs/submission
INDEX="${INDEX:-NeRRve}"                          # name of the index (report, dashboard, file names)
AUTHOR="${AUTHOR:-Koduru Chandrasekhar}"          # name printed on the report cover - edit here
VERSION="${VERSION:-v2 · post-retention build · 18 Nov 2026}"
mkdir -p "$W" "$S"
if [ "$1" != "--skip-models" ]; then
  python -I analysis/01_prepare.py "$DATA" "$W"
  python -I analysis/02_models.py "$W"
fi
python -I analysis/03_ratings.py "$DATA" "$W" 0.75 "" HYBG 300
for spec in "0.75 _vLI LI" "0.75 _vHYBin HYB" "0.75 _v3G HYB3G" "0.65 _vq65 HYBG" "0.85 _vq85 HYBG"; do
  set -- $spec; python -I analysis/03_ratings.py "$DATA" "$W" "$1" "$2" "$3" 0 > /dev/null
done
python -I analysis/06_robustness.py "$W"
python -I analysis/12_curate.py "$W"
python -I analysis/04_insights.py "$DATA" "$W" > "$W/insights.log"
python -I analysis/05_chase_grid.py "$W"
python -I analysis/08_report.py "$W" "$S/${INDEX}_Report.html" "$AUTHOR" "$VERSION" "$INDEX"
python -I analysis/09_render.py "$S/${INDEX}_Report.html" "$S/${INDEX}_Report.pdf"
python -I analysis/10_dashboard.py "$W" "$S/${INDEX}_Dashboard.html" "$VERSION" "$INDEX"
python -I analysis/11_exports.py "$W" "$S" "$INDEX"
echo "ALL DONE"
