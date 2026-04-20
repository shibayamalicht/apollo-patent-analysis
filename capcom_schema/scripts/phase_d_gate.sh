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
# grep -A 50 は最大 51 行しか拾えず、章が 80 行以上あっても誤判定する。
# awk でセクション開始（= クロスモジュール...）から次のレベル1見出し（= XXX）直前までを抽出する。
cross_lines=$(awk '
    /^= .*クロスモジュール/ { start=1; next }
    start && /^= [^=]/ { exit }
    start { print }
' "$REPORT" | wc -l | tr -d ' ')
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

# Check 6: Web 情報の出所記載 — #footnote[ 引用形式に統一
web_use=$(count_matches "市場.*予測|CAGR.*市場|プレスリリース|ニュース|によると|報道" "$REPORT")
footnote_count=$(count_matches '#footnote\[' "$REPORT")
# 廃止済みの旧インライン形式 "_出所:" が残っていないか
legacy_src=$(count_matches '_出所:|_Source:' "$REPORT")
if [ "$legacy_src" -gt 0 ]; then
  echo "❌ Check 6a FAIL: 旧形式のインライン出所 '_出所:' が${legacy_src}件残存 (#footnote[...] に統一する)"
  fail=1
else
  echo "✅ Check 6a OK:  旧形式インライン出所なし"
fi
if [ "$web_use" -gt "$footnote_count" ]; then
  echo "❌ Check 6b FAIL: Web情報使用=${web_use}件 vs #footnote 引用=${footnote_count}件 (脚注化されていない Web 情報あり)"
  fail=1
elif [ "$web_use" -gt 0 ]; then
  echo "✅ Check 6b OK:  Web情報使用=${web_use}件、#footnote 引用=${footnote_count}件"
fi

# Check 7: 仮説と検証のバランス (情報のみ、不合格判定なし)
hyp_count=$(count_matches "仮説|H[1-9]|と推察|可能性がある|と考えられる" "$REPORT")
ver_count=$(count_matches "✅|❌|⚠️|❓|支持|棄却|未検証" "$REPORT")
echo "ℹ️  Check 7: 仮説導出=${hyp_count}件 vs 検証=${ver_count}件 (近い値が望ましい)"

# Check 8: 用語統一チェック（内部識別子がレポートに残っていないか）
# ヒット数が 0 でなければ不合格。terminology.md のルール違反。
json_leak=$(count_matches "(saturnv_clusters|mega_momentum|nebula_[a-z]+|atlas_statistics|explorer_[a-z]+|eagle_[a-z]+|core_classification|crew_network)\.json" "$REPORT")
field_leak=$(count_matches "\b(spatial_context|cluster_dynamics|noise_analysis|quadrant_summary|time_series|hype_cycle_phase|macro_events|npl_gap_analysis)\b" "$REPORT")
mdfile_leak=$(count_matches "(SKILL|common_framework|data_notes|cross_module|deep_dive_guide|report_structure|quality_checklist|terminology|map_reading|patent_citation|executive_summary_guide)\.md" "$REPORT")
# 内部プロセス用語（deep_dive / Phase A-D 等）
process_leak=$(count_matches "\b(deep[-_ ]dive|Phase\s+[ABCD]\b|Phase\s+5a)\b" "$REPORT")

if [ "$json_leak" -gt 0 ]; then
  echo "❌ Check 8a FAIL: 内部JSONファイル名がレポート本文に${json_leak}件残存 (terminology.md 違反)"
  fail=1
else
  echo "✅ Check 8a OK:  内部JSONファイル名の露出なし"
fi
if [ "$field_leak" -gt 0 ]; then
  echo "❌ Check 8b FAIL: 内部フィールド名がレポート本文に${field_leak}件残存 (terminology.md 違反)"
  fail=1
else
  echo "✅ Check 8b OK:  内部フィールド名の露出なし"
fi
if [ "$mdfile_leak" -gt 0 ]; then
  echo "❌ Check 8c FAIL: 内部ガイドファイル名(*.md)がレポート本文に${mdfile_leak}件残存 (terminology.md 違反)"
  fail=1
else
  echo "✅ Check 8c OK:  内部ガイドファイル名の露出なし"
fi
if [ "$process_leak" -gt 0 ]; then
  echo "❌ Check 8d FAIL: 内部プロセス用語(deep_dive / Phase A-D 等)が${process_leak}件残存 (terminology.md 違反)"
  fail=1
else
  echo "✅ Check 8d OK:  内部プロセス用語の露出なし"
fi

# Check 9: J-PlatPat 等の具体DB名がユーザー未指定にも関わらず補われていないか
# population_meta.database_name が未指定なら、執筆者は具体名を勝手に補えない
jplat_leak=$(count_matches "J-PlatPat|JPlatPat" "$REPORT")
db_name=""
if [ -f "voyager/context.json" ]; then
  # population_meta.database_name の有無を軽く確認（jq が無くてもgrepベースで判定）
  db_name=$(grep -A 1 '"database_name"' voyager/context.json 2>/dev/null | grep -oE ':\s*"[^"]*"' | head -n 1 | sed 's/^: *"\(.*\)"$/\1/' || true)
fi
if [ "$jplat_leak" -gt 0 ] && [ -z "$db_name" ]; then
  echo "❌ Check 9 FAIL: J-PlatPat がレポートに${jplat_leak}件出現しているが、population_meta.database_name は未指定"
  echo "   → 執筆者が具体名を勝手に補わないこと。「提供された特許データセット」に置換してください"
  fail=1
elif [ "$jplat_leak" -gt 0 ]; then
  echo "ℹ️  Check 9: J-PlatPat が${jplat_leak}件出現。database_name='${db_name}' と整合しているか目視確認してください"
else
  echo "✅ Check 9 OK:  J-PlatPat の不正混入なし"
fi

# Check 10: スコープ限定ルール (terminology.md §6)
# 「本母集団内で言えること」を「業界全体で言えること」のように読ませる記述の検出。
#
# 判定ロジック:
#   - 無限化語（「業界では」「市場では」「全体として」等）の出現数を数える
#   - 限定語（「本母集団」「本分析の」「本対象特許群」等）の出現数を数える
#   - 限定語が十分多ければ合格 (無限化語 ≤ 限定語 × 0.3)
#
# 「業界」「市場」単独だと頻出しすぎるので、「〜では」「〜の」などの限定助詞との組み合わせで検出する。
# 外部裏付け (#footnote) 付きの一般化は合法だが、このチェックでは区別せず、
# 限定語の多さで全体バランスを判定する。
#
# 例外: 「本分析の前提」章でレポート自身のスコープ注記として無限化語を
# 説明目的で使うケースがあるため、若干の余裕を持たせる (0.3 倍まで許容)。
echo ""
echo "--- Check 10: スコープ限定ルール (本母集団 vs 業界全体の区別) ---"

# 無限化語 (外部裏付けなしに使うと誤読誘発)
# 「本分析の範囲と限界」章でスコープ説明として使われるのを除くため、単なる出現数で判定
unscoped_count=$(count_matches "業界では|業界全体|市場では|市場全体|産業では|全体として|業界の主流|業界の標準|主流技術|業界の|市場の" "$REPORT")

# 限定語 (本母集団内の観察であることを明示)
scoped_count=$(count_matches "本母集団|本分析の特許群|本対象特許群|本検索式|本データセット|本分析の対象|本分析では|本分析において|本母集団に" "$REPORT")

# 判定: 無限化語 > 限定語 × 0.3 ならスコープ明示が不足
threshold_numerator=$((scoped_count * 3))
threshold_check=$((unscoped_count * 10))

echo "ℹ️  Check 10 数値: 無限化語=${unscoped_count}件、限定語=${scoped_count}件"

if [ "$scoped_count" -lt 5 ]; then
  echo "❌ Check 10a FAIL: 限定語 (本母集団/本分析の等) が ${scoped_count}件 しかない (要 5件以上)"
  echo "   → レポート全体で母集団スコープの明示が不足しています"
  echo "   → 各モジュール章・エグゼクティブサマリー・結論章で「本母集団では〜」等の限定修飾を追加してください"
  echo "   → 参照: analysis/terminology.md §6, analysis/common_framework.md §1 各 Layer のスコープ明示ルール"
  fail=1
else
  echo "✅ Check 10a OK:  限定語 = ${scoped_count}件 (要 5件以上)"
fi

# 無限化語 > 限定語 × 0.3 なら警告
# 整数比較のため「無限化語 × 10 > 限定語 × 3」で判定（= 無限化語/限定語 > 0.3）
if [ "$unscoped_count" -gt 0 ] && [ "$threshold_check" -gt "$threshold_numerator" ]; then
  echo "❌ Check 10b FAIL: 無限化語=${unscoped_count}件 が 限定語=${scoped_count}件 の 0.3 倍を超過"
  echo "   → 「業界では」「市場では」「全体として」等が多用されています"
  echo "   → 「本母集団では〜」「本分析の特許群では〜」等に置換するか、"
  echo "     外部裏付け (#footnote[...]) を必ず付してください"
  echo "   → 参照: analysis/terminology.md §6 スコープ限定ルール"
  fail=1
elif [ "$unscoped_count" -gt 0 ]; then
  echo "✅ Check 10b OK:  無限化語=${unscoped_count}件 は 限定語=${scoped_count}件 の 0.3 倍以内"
else
  echo "✅ Check 10b OK:  無限化語の出現なし"
fi

# Check 11: 母集団タイプ別の不適切表現検出 (population_type_metrics.md §3)
# Phase A STOP-GATE C で確定した母集団タイプを reports/_phase_a_decisions.json から読み取り、
# タイプ B/C/D の場合、市場・業界解釈の表現が混入していないかチェックする。
#
# ロジック:
#   1. reports/_phase_a_decisions.json が存在する場合のみ実施（存在しなければスキップ + 警告）
#   2. population_type.code を抽出
#   3. タイプ B/C/D の場合: forbidden_expressions リストに基づき本文を検査
#   4. タイプ A/A' の場合: 特に追加チェックなし（Check 10 のスコープ限定で既にカバー）
echo ""
echo "--- Check 11: 母集団タイプ別の不適切表現検出 (population_type_metrics.md §3) ---"

DECISIONS_FILE="reports/_phase_a_decisions.json"
if [ ! -f "$DECISIONS_FILE" ]; then
  echo "⚠️  Check 11 WARN: $DECISIONS_FILE が存在しません"
  echo "   → Phase A STOP-GATE C で母集団タイプ判定が実施されていない可能性があります"
  echo "   → SKILL.md Phase A STOP-GATE C の手順に従って判定し、"
  echo "     reports/_phase_a_decisions.json を作成してください"
  echo "   → 本チェックは判定ファイルがある場合のみ実施されます (FAIL はしないが強く推奨)"
else
  # population_type.code を抽出 (jq が無くても grep ベースで取れる)
  pop_type=$(grep -o '"code"[[:space:]]*:[[:space:]]*"[^"]*"' "$DECISIONS_FILE" 2>/dev/null | head -n 1 | sed 's/.*"code"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

  if [ -z "$pop_type" ]; then
    echo "⚠️  Check 11 WARN: $DECISIONS_FILE から population_type.code を取得できません"
    echo "   → ファイル形式を確認してください"
  else
    echo "ℹ️  Check 11: 母集団タイプ = ${pop_type}"

    case "$pop_type" in
      B|C|D)
        # タイプ B/C/D 共通の市場・業界解釈表現を検出
        # これらはタイプ B/C/D では誤読を誘発するため禁止
        market_terms=$(count_matches "市場は寡占|市場集中|市場構造|市場シェア|業界シェア|業界の集中度|業界の集中|競争環境は|業界全体で|業界の主要|業界のトレンド|業界動向|業界全体の出願" "$REPORT")

        # タイプ C (単一企業) 追加禁止: 競合分析, 業界, 市場
        # これらは自社分析の母集団では完全に誤読
        type_c_extra_terms=0
        if [ "$pop_type" = "C" ]; then
          type_c_extra_terms=$(count_matches "競合分析|市場動向|業界動向|業界環境|市場環境" "$REPORT")
        fi

        total_bad=$((market_terms + type_c_extra_terms))

        if [ "$total_bad" -gt 0 ]; then
          echo "❌ Check 11 FAIL: タイプ ${pop_type} で禁止される市場・業界解釈表現が ${total_bad}件 検出されました"
          echo "   → タイプ ${pop_type} では「市場は寡占」「業界シェア」「市場構造」等は誤読を誘発します"
          echo "   → タイプ B: 「調査対象 N 社内での〜」に置換"
          echo "   → タイプ C: 「当社の〜」に置換 (発明者集中度・IPC 集中度で代替)"
          echo "   → タイプ D: 「本テーマでは〜」に置換"
          echo "   → 詳細: analysis/population_type_metrics.md §3 (タイプ別禁止表現リスト)"
          fail=1
        else
          echo "✅ Check 11 OK:  タイプ ${pop_type} でも市場・業界解釈表現なし"
        fi

        # タイプ C で出願人 HHI の言及があるかチェック (単一企業では HHI=1.0 で無意味)
        if [ "$pop_type" = "C" ]; then
          applicant_hhi=$(count_matches "出願人.*HHI|HHI.*出願人|出願人集中度" "$REPORT")
          if [ "$applicant_hhi" -gt 0 ]; then
            echo "❌ Check 11 (C 追加) FAIL: タイプ C (単一企業) で出願人 HHI の言及が ${applicant_hhi}件"
            echo "   → 単一企業母集団では HHI = 1.0 で無意味です"
            echo "   → 発明者集中度・IPC 集中度に置換してください"
            fail=1
          else
            echo "✅ Check 11 (C 追加) OK:  タイプ C で出願人 HHI の不適切言及なし"
          fi
        fi
        ;;

      A|A\'|Aprime)
        # タイプ A / A' は Check 10 のスコープ限定ルールで既にカバー
        echo "✅ Check 11: タイプ ${pop_type} (業界全体/技術領域) のため、Check 10 のスコープ限定ルールで保護されます"
        ;;

      *)
        echo "⚠️  Check 11 WARN: 未知のタイプコード '${pop_type}' (A/A'/B/C/D のいずれかであるべき)"
        ;;
    esac
  fi
fi

# Check 12: 設計意図 (query_intent) の一貫性 (terminology.md §5-A、report_structure.md、deep_dive_guide.md)
# 3 つのサブチェック:
#   12a: 意図参照語のカウント (5 章 × 1 箇所以上なので最低 5 件期待)
#   12b: 問い/答え形式 (Q1/A1/SQ1 等) の禁止
#   12c: _phase_a_decisions.json の sub_questions.keywords が結論章でカバーされているか
echo ""
echo "--- Check 12: 設計意図 (query_intent) の一貫性 (terminology.md §5-A) ---"

# Check 12a: 意図参照語のカウント
# 「本分析の視座」「設計意図」「当初の分析観点」「本分析の観点」「意図に照らす」「視座に照らす」「分析の視座」
intent_refs=$(count_matches "本分析の視座|設計意図|当初の分析観点|当初の分析観点|当初意図|本分析の観点|意図に照らす|視座に照らす|分析の視座" "$REPORT")
echo "ℹ️  Check 12a 数値: 意図参照語 = ${intent_refs}件"

if [ "$intent_refs" -lt 5 ]; then
  echo "❌ Check 12a FAIL: 意図参照語 = ${intent_refs}件 (要 5件以上)"
  echo "   → 以下 5 章で「本分析の視座」等の意図参照を最低 1 箇所ずつ含めてください:"
  echo "     (1) エグゼクティブサマリー冒頭"
  echo "     (2) 各 deep_dive 冒頭"
  echo "     (3) クロスモジュール分析の結論段"
  echo "     (4) 戦略的提言"
  echo "     (5) 仮説検証サマリー"
  echo "   → 参照: analysis/terminology.md §5-A-1 (指定章での意図参照義務化)"
  fail=1
else
  echo "✅ Check 12a OK:  意図参照語 = ${intent_refs}件 (要 5件以上)"
fi

# Check 12b: 問い/答え形式の禁止
# レポート本文に Q1/A1/SQ1 等の記号や「サブクエスチョン」「問い 1」等の表現が出現したら FAIL
forbidden_qa=$(count_matches "^Q[0-9]+[:：\.]|^A[0-9]+[:：\.]|SQ[0-9]+|サブクエスチョン|問い\s*[0-9]+|第\s*[0-9]+\s*の問い" "$REPORT")

if [ "$forbidden_qa" -gt 0 ]; then
  echo "❌ Check 12b FAIL: 問い/答え形式の記号・表現が ${forbidden_qa}件 検出されました"
  echo "   → レポート本文に「Q1」「A1」「SQ1」「問い 1」「サブクエスチョン」等の記号・表現は禁止"
  echo "   → 通常の宣言調の論述で書いてください"
  echo "   → 例: 「本分析の視座に即して答えると、〜が明らかとなった」"
  echo "   → 参照: analysis/terminology.md §5-A-2 (サブクエスチョン化・問い形式禁止)"
  fail=1
else
  echo "✅ Check 12b OK:  問い/答え形式の混入なし"
fi

# Check 12c: sub_questions の keywords が結論章でカバーされているか
DECISIONS_FILE="reports/_phase_a_decisions.json"
if [ -f "$DECISIONS_FILE" ]; then
  # _phase_a_decisions.json から keywords をフラット抽出
  # "keywords": ["a", "b", ...] → a, b, ... を行単位で出力
  # 複数行にわたる JSON を考慮して tr で改行除去後に抽出
  keywords_raw=$(tr -d '\n' < "$DECISIONS_FILE" | grep -oE '"keywords"[[:space:]]*:[[:space:]]*\[[^]]*\]' | grep -oE '"[^"]*"' | grep -vE '^"keywords"$' | sed 's/^"//; s/"$//')

  if [ -n "$keywords_raw" ]; then
    # 結論章（戦略的提言以降）を抽出
    # SKILL.md で定義された章構成では、戦略的提言 → 付録 の順
    conclusion_text=$(awk '
        /^= .*戦略的提言|^= .*戦略的インプリケーション/ { start=1 }
        start && /^= 付録/ { exit }
        start { print }
    ' "$REPORT")

    missing_keywords=""
    total_keywords=0
    found_keywords=0

    # while read で keywords を 1 つずつ処理
    while IFS= read -r kw; do
      if [ -z "$kw" ]; then continue; fi
      total_keywords=$((total_keywords + 1))
      if echo "$conclusion_text" | grep -q -F "$kw"; then
        found_keywords=$((found_keywords + 1))
      else
        if [ -z "$missing_keywords" ]; then
          missing_keywords="$kw"
        else
          missing_keywords="$missing_keywords, $kw"
        fi
      fi
    done <<EOF
$keywords_raw
EOF

    if [ "$total_keywords" -eq 0 ]; then
      echo "ℹ️  Check 12c: _phase_a_decisions.json に sub_questions.keywords がないためスキップ"
    elif [ -n "$missing_keywords" ]; then
      echo "❌ Check 12c FAIL: sub_questions のキーワード ${total_keywords}件 中 ${found_keywords}件のみ結論章で登場"
      echo "   → 結論章（戦略的提言セクション以降）に登場していないキーワード: ${missing_keywords}"
      echo "   → 結論章（総括 + 戦略的インプリケーション + 推奨アクション）でこれらの観点を網羅してください"
      echo "   → 参照: analysis/terminology.md §5-A-2 (サブクエスチョン化)、analysis/report_structure.md §4 (戦略的提言)"
      fail=1
    else
      echo "✅ Check 12c OK:  sub_questions のキーワード ${total_keywords}件 すべて結論章で登場"
    fi
  else
    echo "ℹ️  Check 12c: sub_questions.keywords が空のためスキップ"
  fi
else
  echo "⚠️  Check 12c WARN: $DECISIONS_FILE が存在しません (Phase A サブクエスチョン化未実施)"
  echo "   → SKILL.md Phase A サブクエスチョン化 STOP-GATE の手順に従って判定・保存してください"
fi

# Check 13: NEBULA 戦略の検証 (population_type_metrics.md §4-3 nebula_strategy)
# Phase A STOP-GATE D で確定した nebula_strategy に応じて、レポートが正しく対応しているかを検証。
#
# モード別チェック:
#   execute: NEBULA 章があるか (nebula_deep_dive.typ または本文中に NEBULA 章タイトル)
#   web_compensation: 「外部環境分析」章があり、4 カテゴリすべてが #footnote[...] 付きでカバーされているか
#   omit: 「本分析の範囲と限界」章に特許情報のみ対象の注記があり、NEBULA 章が存在しないこと
echo ""
echo "--- Check 13: NEBULA 戦略の検証 (population_type_metrics.md §4-3) ---"

if [ ! -f "$DECISIONS_FILE" ]; then
  echo "⚠️  Check 13 WARN: $DECISIONS_FILE が存在しません (nebula_strategy 未判定)"
  echo "   → Phase A STOP-GATE D の手順に従って判定・保存してください"
else
  # nebula_strategy.selected_mode を抽出
  nebula_mode=$(tr -d '\n' < "$DECISIONS_FILE" | grep -oE '"nebula_strategy"[^}]*"selected_mode"[[:space:]]*:[[:space:]]*"[^"]*"' | grep -oE '"selected_mode"[[:space:]]*:[[:space:]]*"[^"]*"' | sed 's/.*"selected_mode"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/')

  if [ -z "$nebula_mode" ]; then
    echo "⚠️  Check 13 WARN: $DECISIONS_FILE から nebula_strategy.selected_mode を取得できません"
    echo "   → Phase A STOP-GATE D が未完了の可能性があります"
  else
    echo "ℹ️  Check 13: NEBULA 戦略モード = ${nebula_mode}"

    case "$nebula_mode" in
      execute)
        # execute モード: NEBULA 章があるかチェック
        nebula_chapter=$(count_matches "^= NEBULA|^= 環境分析|nebula_deep_dive" "$REPORT")
        if [ "$nebula_chapter" -lt 1 ]; then
          echo "❌ Check 13 FAIL: モード execute だが NEBULA 章が本文に見当たりません"
          echo "   → reports/nebula_deep_dive.typ を生成し、report.typ に統合してください"
          fail=1
        else
          echo "✅ Check 13 OK:  モード execute で NEBULA 章を検出"
        fi
        ;;

      web_compensation)
        # web_compensation モード: 「外部環境分析」章があり、4 カテゴリすべてが #footnote 付きでカバー
        external_chapter=$(count_matches "外部環境分析|外部環境" "$REPORT")
        if [ "$external_chapter" -lt 1 ]; then
          echo "❌ Check 13 FAIL: モード web_compensation だが「外部環境分析」章が見当たりません"
          echo "   → 4 カテゴリ（市場規模・政策・学術動向・主要企業動向）を Web 調査で補完した章を追加してください"
          echo "   → 参照: analysis/deep_dive_guide.md Step 0（web_compensation モード節）"
          fail=1
        else
          # 4 カテゴリが本文にカバーされているか
          cat_market=$(count_matches "市場規模|業界統計|市場予測" "$REPORT")
          cat_policy=$(count_matches "政策|規制|標準化" "$REPORT")
          cat_academic=$(count_matches "学術|論文|研究機関|キーパーソン" "$REPORT")
          cat_corporate=$(count_matches "主要企業|プレスリリース|M&A|事業戦略" "$REPORT")

          missing=""
          [ "$cat_market" -lt 1 ] && missing="${missing}市場規模 "
          [ "$cat_policy" -lt 1 ] && missing="${missing}政策・規制 "
          [ "$cat_academic" -lt 1 ] && missing="${missing}学術動向 "
          [ "$cat_corporate" -lt 1 ] && missing="${missing}主要企業動向 "

          if [ -n "$missing" ]; then
            echo "❌ Check 13 FAIL: モード web_compensation で 4 カテゴリのうち以下がカバーされていません: ${missing}"
            echo "   → 各カテゴリにつき最低 1 件の Web 調査結果を #footnote[...] 付きで本文に反映してください"
            fail=1
          else
            # #footnote 件数が 4 以上あるかも確認
            ext_footnotes=$(count_matches '#footnote\[' "$REPORT")
            if [ "$ext_footnotes" -lt 4 ]; then
              echo "❌ Check 13 FAIL: モード web_compensation で #footnote[...] 引用が ${ext_footnotes}件 (4 カテゴリ分・最低 4 件必要)"
              fail=1
            else
              echo "✅ Check 13 OK:  モード web_compensation で 4 カテゴリすべて + #footnote=${ext_footnotes}件 を検出"
            fi
          fi
        fi
        ;;

      omit)
        # omit モード: NEBULA 章がなく、範囲と限界に特許のみ対象の注記があるか
        nebula_chapter=$(count_matches "^= NEBULA|nebula_deep_dive" "$REPORT")
        if [ "$nebula_chapter" -gt 0 ]; then
          echo "❌ Check 13 FAIL: モード omit だが NEBULA 章が ${nebula_chapter}件 残存"
          echo "   → 省略モードでは NEBULA 章を削除してください"
          fail=1
        fi

        # 「本分析の範囲と限界」章に特許情報のみ対象の注記があるか
        scope_note=$(count_matches "特許情報のみ|特許のみを対象|外部環境データは.*対象外" "$REPORT")
        if [ "$scope_note" -lt 1 ]; then
          echo "❌ Check 13 FAIL: モード omit だが「本分析の範囲と限界」章に特許情報のみ対象の注記がありません"
          echo "   → 「本分析は特許情報のみを対象とし、外部環境データは取り扱わない」旨を 2-3 行で明記してください"
          echo "   → 参照: analysis/report_structure.md §3（本分析の範囲と限界テンプレート）"
          fail=1
        fi

        if [ "$nebula_chapter" -eq 0 ] && [ "$scope_note" -ge 1 ]; then
          echo "✅ Check 13 OK:  モード omit で NEBULA 章なし + 範囲と限界の注記あり"
        fi
        ;;

      *)
        echo "⚠️  Check 13 WARN: 未知の nebula_strategy.selected_mode '${nebula_mode}' (execute/web_compensation/omit のいずれかであるべき)"
        ;;
    esac
  fi
fi

echo ""
if [ $fail -eq 1 ]; then
  echo "🛑 Phase D GATE FAILED. quality_checklist.md / terminology.md の不合格項目を修正してください。"
  echo "   (SKILL.md ## 0. 絶対遵守ゲートルール 第3項: 質的判断で量的基準を上書きしない)"
  exit 1
fi

echo "✅ Phase D GATE PASSED. レポート完成。"
exit 0
