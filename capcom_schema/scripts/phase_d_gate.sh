#!/usr/bin/env bash
# Phase D 品質検証ゲートスクリプト
# 使い方: cd <session_dir> && bash capcom_schema/scripts/phase_d_gate.sh
# analysis/quality_checklist.md section 1 の定量チェックコマンドを統合実行する。
#
# このスクリプトは SKILL.md ## 0. 絶対遵守ゲートルール 第3項
# 「不合格時は強制ループ」を機械的に保証するためのゲートである。
# 「自前のチェックで代替」は禁止 (再現性のないチェックは無効)。

set -o pipefail
# Note: -u (unbound variable) は使わない (macOS bash 3.2 互換のため)

# grep -c の安全ラッパ (macOS の古い grep は 0件マッチで exit 1 → 必ず数値で返す)
count_matches() {
  local pattern="$1"
  local file="$2"
  local n
  n=$(grep -c -E "$pattern" "$file" 2>/dev/null || true)
  # 空文字や複数行を 0 に正規化
  n=$(echo "$n" | head -n 1 | tr -d ' \n')
  echo "${n:-0}"
}

REPORT="reports/report.typ"
fail=0
echo "=== Phase D 品質ゲート判定 ==="
echo "Target: $REPORT"
echo ""

if [ ! -f "$REPORT" ]; then
  echo "❌ MISSING: $REPORT が存在しません"
  exit 1
fi

# Check 1: 行数
lines=$(wc -l < "$REPORT" | tr -d ' ')
if [ "$lines" -lt 800 ]; then
  echo "❌ Check 1 FAIL: report.typ = ${lines}行 (要 800行以上、内容不足)"
  fail=1
else
  echo "✅ Check 1 OK:   report.typ = ${lines}行 (要 800行以上)"
fi

# Check 2: 代表特許引用数
patent_cites=$(count_matches "特開|特許第|WO20|JP20" "$REPORT")
if [ "$patent_cites" -lt 15 ]; then
  echo "❌ Check 2 FAIL: 代表特許引用 = ${patent_cites}件 (要 15件以上)"
  fail=1
else
  echo "✅ Check 2 OK:   代表特許引用 = ${patent_cites}件 (要 15件以上)"
fi

# Check 3: 4層モデルキーワード
layer2=$(count_matches "解釈|示唆|と解釈" "$REPORT")
layer3=$(count_matches "にもかかわらず|と合わせて|洞察|統合的に" "$REPORT")
layer4=$(count_matches "検討すべき|推奨|を検討|参入|投資" "$REPORT")
echo "ℹ️  Check 3: Layer2(解釈)=${layer2}, Layer3(洞察)=${layer3}, Layer4(提言)=${layer4} (各5件以上が目安)"
for n in $layer2 $layer3 $layer4; do
  if [ "$n" -lt 5 ]; then
    echo "⚠️  Check 3 WARN: 4層モデルのいずれかが5件未満"
    break
  fi
done

# Check 4: クロスモジュール分析の分量
cross_lines=$(grep -A 50 "クロスモジュール統合分析" "$REPORT" | wc -l | tr -d ' ')
if [ "$cross_lines" -lt 80 ]; then
  echo "❌ Check 4 FAIL: クロスモジュール統合分析 = ${cross_lines}行 (要 80行以上、各パターン15-20行 x 最低3パターン)"
  fail=1
else
  echo "✅ Check 4 OK:   クロスモジュール統合分析 = ${cross_lines}行 (要 80行以上)"
fi

# Check 5: snapshot-figure 数
fig_count=$(count_matches "snapshot-figure" "$REPORT")
if [ "$fig_count" -lt 8 ]; then
  echo "❌ Check 5 FAIL: snapshot-figure = ${fig_count}枚 (要 8枚以上)"
  fail=1
else
  echo "✅ Check 5 OK:   snapshot-figure = ${fig_count}枚 (要 8枚以上)"
fi

# Check 6: Web 情報の出所記載 (Web 情報がある場合のみ警告)
web_use=$(count_matches "市場.*予測|CAGR.*市場|プレスリリース|ニュース|によると|報道" "$REPORT")
web_src=$(count_matches "出所:|footnote|取得日:" "$REPORT")
if [ "$web_use" -gt "$web_src" ]; then
  echo "❌ Check 6 FAIL: Web情報使用=${web_use}件 vs 出所記載=${web_src}件 (出所のないWeb情報あり)"
  fail=1
elif [ "$web_use" -gt 0 ]; then
  echo "✅ Check 6 OK:   Web情報使用=${web_use}件、出所記載=${web_src}件"
fi

# Check 7: 仮説と検証のバランス (情報のみ、不合格判定なし)
hyp_count=$(count_matches "仮説|H[1-9]|と推察|可能性がある|と考えられる" "$REPORT")
ver_count=$(count_matches "✅|❌|⚠️|❓|支持|棄却|未検証" "$REPORT")
echo "ℹ️  Check 7: 仮説導出=${hyp_count}件 vs 検証=${ver_count}件 (近い値が望ましい)"

echo ""
if [ $fail -eq 1 ]; then
  echo "🛑 Phase D GATE FAILED. quality_checklist.md の不合格項目を修正してください。"
  echo "   (SKILL.md ## 0. 絶対遵守ゲートルール 第3項: 質的判断で量的基準を上書きしない)"
  exit 1
fi

echo "✅ Phase D GATE PASSED. レポート完成。"
exit 0
