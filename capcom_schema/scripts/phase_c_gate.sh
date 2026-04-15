#!/usr/bin/env bash
# Phase C ゲート判定スクリプト
# 使い方: cd <session_dir> && bash capcom_schema/scripts/phase_c_gate.sh
# 各 deep_dive ファイルの存在と最低行数を客観的にチェックする
#
# このスクリプトは SKILL.md ## 0. 絶対遵守ゲートルール 第3項
# 「不合格時は強制ループ」を機械的に保証するためのゲートである。

set -o pipefail
# Note: -u (unbound variable) は使わない (macOS bash 3.2 互換のため)

# モジュール最低行数を取得 (SKILL.md L142-152 と整合)
# bash 3.2 互換のため case 文で表現 (連想配列 declare -A は bash 4+ 必要)
get_required() {
  case "$1" in
    nebula)   echo 120 ;;
    saturnv)  echo 250 ;;
    explorer) echo 200 ;;
    mega)     echo 120 ;;
    atlas)    echo 120 ;;
    core)     echo  80 ;;
    *)        echo   0 ;;
  esac
}
# CREW は最低行数指定なし(SKILL.md L152)、ファイル存在だけ確認

REPORTS_DIR="reports"
fail=0
echo "=== Phase C ゲート判定 ==="
echo "Reports dir: $REPORTS_DIR"
echo ""

for module in nebula saturnv explorer mega atlas core; do
  file="$REPORTS_DIR/${module}_deep_dive.typ"
  required=$(get_required "$module")

  if [ ! -f "$file" ]; then
    echo "❌ MISSING: $file (required: ${required}行以上)"
    fail=1
    continue
  fi

  lines=$(wc -l < "$file" | tr -d ' ')

  if [ "$lines" -lt "$required" ]; then
    echo "❌ FAIL: $file = ${lines}行 (要 ${required}行以上、不足 $((required - lines))行)"
    fail=1
  else
    echo "✅ OK:   $file = ${lines}行 (要 ${required}行)"
  fi
done

# CREW (任意モジュール、最低行数指定なし)
crew_file="$REPORTS_DIR/crew_deep_dive.typ"
if [ -f "$crew_file" ]; then
  crew_lines=$(wc -l < "$crew_file" | tr -d ' ')
  echo "ℹ️  INFO: $crew_file = ${crew_lines}行 (CREW は最低行数指定なし)"
fi

echo ""
if [ $fail -eq 1 ]; then
  echo "🛑 Phase C GATE FAILED. 不足を補強してから再実行してください。"
  echo "   (SKILL.md ## 0. 絶対遵守ゲートルール 第3項: 質的判断で量的基準を上書きしない)"
  exit 1
fi

echo "✅ Phase C GATE PASSED. Phase D に進めます。"
exit 0
