#!/usr/bin/env bash
# Phase C ゲート判定スクリプト
# 使い方: cd <session_dir> && bash capcom_schema/scripts/phase_c_gate.sh
# 各 deep_dive ファイルの存在と最低行数を客観的にチェックする
#
# このスクリプトは SKILL.md ## 0. 絶対遵守ゲートルール 第3項
# 「不合格時は強制ループ」を機械的に保証するためのゲートである。

set -o pipefail
# Note: -u (unbound variable) は使わない (macOS bash 3.2 互換のため)

# モジュール最低「文字数（非空白）」を取得。
# 旧版は wc -l（行数）で判定していたが、1文ずつ改行すれば行数を水増しできる
# （Typst では改行は描画上スペースなので、見た目は同じまま行数だけ膨らむ）。
# 合否は改行で増やせない「文字数」で判定する。行数は参考表示のみ。
# bash 3.2 互換のため case 文で表現 (連想配列 declare -A は bash 4+ 必要)
get_required_chars() {
  case "$1" in
    nebula)   echo  6500 ;;
    saturnv)  echo 13000 ;;
    explorer) echo 10500 ;;
    mega)     echo  6500 ;;
    atlas)    echo  6500 ;;
    core)     echo  4400 ;;
    *)        echo     0 ;;
  esac
}
# CREW は最低量の指定なし、ファイル存在だけ確認

REPORTS_DIR="reports"
fail=0
echo "=== Phase C ゲート判定 ==="
echo "Reports dir: $REPORTS_DIR"
echo "判定基準: 非空白文字数（行数は1文1行で水増し可能なため参考値）"
echo ""

for module in nebula saturnv explorer mega atlas core; do
  file="$REPORTS_DIR/${module}_deep_dive.typ"
  required=$(get_required_chars "$module")

  if [ ! -f "$file" ]; then
    echo "❌ MISSING: $file (要 ${required}字以上)"
    fail=1
    continue
  fi

  # 非空白文字数と「1文1行率」を python で測る（行数は水増し可能なので文字数で判定）
  stats=$(python3 - "$file" <<'PYEOF'
import sys, re
raw = open(sys.argv[1], encoding='utf-8').read()
chars = len(re.sub(r'\s', '', raw))
body = [l for l in raw.split('\n') if l.strip() and not l.strip().lstrip().startswith(('//', '#', '='))]
one = sum(1 for l in body if l.rstrip().endswith('。'))
ratio = (one * 100) // len(body) if body else 0
print("%d %d" % (chars, ratio))
PYEOF
)
  chars=$(echo "$stats" | awk '{print $1}')
  onesent=$(echo "$stats" | awk '{print $2}')
  lines=$(wc -l < "$file" | tr -d ' ')

  if [ "$chars" -lt "$required" ]; then
    echo "❌ FAIL: $file = ${chars}字 (要 ${required}字以上・不足 $((required - chars))字／行数=${lines}・1文1行率=${onesent}%)"
    echo "   → 行数は1文ずつ改行すると水増しできるため文字数で判定。発展した段落（主張→根拠→示唆）で内容を増やすこと"
    fail=1
  else
    echo "✅ OK:   $file = ${chars}字 (要 ${required}字・行数=${lines})"
    if [ "$onesent" -ge 92 ]; then
      echo "   ⚠️ 1文1行率=${onesent}%（羅列調・水増しの兆候）。複数文で論証を組む段落を増やすと読み物として良くなる"
    fi
  fi
done

# CREW (任意モジュール、最低行数指定なし)
crew_file="$REPORTS_DIR/crew_deep_dive.typ"
if [ -f "$crew_file" ]; then
  crew_lines=$(wc -l < "$crew_file" | tr -d ' ')
  echo "ℹ️  INFO: $crew_file = ${crew_lines}行 (CREW は最低行数指定なし)"
fi

# 本文のスクリプト機械生成の早期検出（Phase D Check 19a と同型・FAIL）
# deep_dive.typ をテンプレート生成するのは水増しの温床。Phase C の段階で止める。
gen_scripts=$(find "$REPORTS_DIR" -maxdepth 1 -name '*.py' 2>/dev/null | while read -r f; do
  if grep -qE "\.typ" "$f" 2>/dev/null; then echo "$f"; fi
done)
if [ -n "$gen_scripts" ]; then
  echo "❌ FAIL: deep_dive を生成する Python スクリプトを検出（.py が .typ を書き出している）"
  echo "$gen_scripts" | sed 's/^/     /'
  echo "   → 本文はスクリプトでテンプレート生成しない。各文を固有の分析として直接書く（SKILL.md §0 第7項）"
  fail=1
fi

# 工程ナレーション節・後続章への申し送りの早期検出（Phase D Check 8e と同型・FAIL）
# 「後続分析への接続」等のメタ節や「○○分析では〜を確認する」型の後続章ToDoは分析ではなく水増し。
narration_hits=$(grep -rnE "(後続分析|後続章|後続の(分析|章|モジュール|CORE|Saturn|MEGA|Explorer|CREW|EAGLE|ATLAS|NEBULA)|次章への|分析への申し送り|今後の分析の方向)" "$REPORTS_DIR"/*_deep_dive.typ 2>/dev/null | head -10)
if [ -n "$narration_hits" ]; then
  echo "❌ FAIL: 工程ナレーション/後続章への申し送りを検出（「後続分析への接続」等のメタ節・申し送り＝水増し）"
  echo "$narration_hits" | sed 's/^/     /'
  echo "   → 各章は自章の結論で閉じ、章間連携は『クロスモジュール統合分析』章で行う。他章参照は「〜で確認された〈事実〉」の過去形に限る（SKILL.md §0 第8項・terminology.md §1-4）"
  fail=1
fi

# フェーズ間引き継ぎ日誌の存在チェック（ソフト・WARN のみ・exit は落とさない）
# 分割実行（別スレッド/コンテキスト枯渇）で「分析の記憶」を失わないためのアンカー。
if [ ! -f "$REPORTS_DIR/_carryover.md" ]; then
  echo "⚠️  WARN: $REPORTS_DIR/_carryover.md が未作成（フェーズ分割すると分析の記憶が失われやすい）"
  echo "   → capcom_schema/templates/carryover_template.md を reports/_carryover.md にコピーし、各フェーズ完了時に更新する"
fi

echo ""
if [ $fail -eq 1 ]; then
  echo "🛑 Phase C GATE FAILED. 不足を補強してから再実行してください。"
  echo "   (SKILL.md ## 0. 絶対遵守ゲートルール 第3項: 質的判断で量的基準を上書きしない)"
  exit 1
fi

echo "✅ Phase C GATE PASSED. Phase D に進めます。"
exit 0
