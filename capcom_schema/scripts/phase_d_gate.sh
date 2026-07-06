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

# Check 1: 内容量（非空白文字数で判定。行数は1文1行で水増し可能なため参考値）
lines=$(wc -l < "$REPORT" | tr -d ' ')
report_chars=$(python3 -c "import re,sys; print(len(re.sub(r'\s','',open(sys.argv[1],encoding='utf-8').read())))" "$REPORT" 2>/dev/null || echo 0)
if [ "$report_chars" -lt 45000 ]; then
  echo "❌ Check 1 FAIL: report.typ = ${report_chars}字 (要 45000字以上、内容不足／行数=${lines})"
  echo "   → 行数(wc -l)は1文ずつ改行すると水増しできるため文字数で判定する。発展した段落で内容を増やすこと"
  echo "   → 文字数不足は固有事実で埋める（自明な一般論で埋めない）。新しい代表特許(固有の公開番号)・数値根拠・別クロスパターン・Web裏付けを足す"
  fail=1
else
  echo "✅ Check 1 OK:   report.typ = ${report_chars}字 (要 45000字以上・行数=${lines})"
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
layer3=$(count_matches "にもかかわらず|と合わせて|洞察|統合的に|一見|逆説|二面性|裏を返せば|照合すると|突き合わせると|総合すると" "$REPORT")
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
if [ "$cross_lines" -lt 120 ]; then
  echo "❌ Check 4 FAIL: クロスモジュール統合分析 = ${cross_lines}行 (要 120行以上、各パターン15-20行 x 最低5パターン)"
  fail=1
else
  echo "✅ Check 4 OK:   クロスモジュール統合分析 = ${cross_lines}行 (要 120行以上)"
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
# Check 6c: Web 出所のプレースホルダ URL 残存検出（WARN・data_notes.md §3 必須3要素）
#   exemplar/ガイドの (URL) や https://….com/xxx・/... をコピペで残すと「URL未記載」と同義。
#   サイト名・URL・取得日の3要素は揃って初めて検証可能な出所になる。
url_placeholder=$(count_matches '\(URL\)|（URL）|/xxx|/XXX|/\.\.\.|https?://[^ )）]*\.\.\.' "$REPORT")
if [ "$url_placeholder" -gt 0 ]; then
  echo "⚠️  Check 6c WARN: Web出所にプレースホルダURL（(URL) / /xxx / /... 等）が ${url_placeholder}件。脚注・付録Cの URL は取得した実際の完全URLに置換する（data_notes.md §3 必須3要素: サイト名・URL・取得日）"
  grep -nE '\(URL\)|（URL）|/xxx|/XXX|/\.\.\.|https?://[^ )）]*\.\.\.' "$REPORT" 2>/dev/null | head -5 | sed 's/^/   該当: /'
else
  echo "✅ Check 6c OK:  Web出所URLにプレースホルダなし"
fi

# Check 7: 仮説と検証のバランス (情報のみ、不合格判定なし)
hyp_count=$(count_matches "仮説|H[1-9]|と推察|可能性がある|と考えられる" "$REPORT")
ver_count=$(count_matches "✅|❌|⚠️|❓|支持|棄却|未検証" "$REPORT")
echo "ℹ️  Check 7: 仮説導出=${hyp_count}件 vs 検証=${ver_count}件 (近い値が望ましい)"

# Check 8: 用語統一チェック（内部識別子がレポートに残っていないか）
# ヒット数が 0 でなければ不合格。terminology.md のルール違反。
json_leak=$(count_matches "(saturnv_clusters|mega_momentum[a-z_]*|nebula_[a-z]+|atlas_statistics|explorer_[a-z]+|eagle_[a-z]+|core_classification|crew_network|_phase_a_decisions)\.json" "$REPORT")
field_leak=$(count_matches "\b(spatial_context|cluster_dynamics|noise_analysis|quadrant_summary|time_series|hype_cycle_phase|macro_events|npl_gap_analysis)\b" "$REPORT")
# 内部作業メモ（引き継ぎ日誌 _carryover.md 等）の本文流出も検出する
mdfile_leak=$(count_matches "(SKILL|common_framework|data_notes|cross_module|deep_dive_guide|report_structure|quality_checklist|terminology|map_reading|patent_citation|executive_summary_guide|_carryover)\.md" "$REPORT")
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

# Check 8e: 工程ナレーション節・後続章への申し送りの検出（意味のない水増し節）
#   「後続分析への接続」「次章への申し送り」等のメタ節や、「Explorer分析では…確認する」型の
#   後続章ToDo羅列は分析ではなく工程ナレーション＝水増し。完成レポートでは各章が既に存在するため無価値。
#   章間連携は「クロスモジュール統合分析」章が担う。他章参照は「○○で確認された〈事実〉」の根拠引用に限る。
narration_leak=$(count_matches "(後続分析|後続章|後続の(分析|章|モジュール|CORE|Saturn|MEGA|Explorer|CREW|EAGLE|ATLAS|NEBULA)|次章への|分析への申し送り|今後の分析の方向)" "$REPORT")
meta_heading=$(count_matches "^=+[[:space:]].*(後続分析|後続章|次章への|申し送り|ハンドオフ|今後の分析の方向)" "$REPORT")
if [ "$meta_heading" -gt 0 ] || [ "$narration_leak" -gt 0 ]; then
  echo "❌ Check 8e FAIL: 工程ナレーション/後続章への申し送りを検出 (メタ節見出し=${meta_heading}件・本文=${narration_leak}件)"
  echo "   → 「後続分析への接続」「次章への申し送り」等のメタ節・段落は禁止。各章は自章の結論で閉じ、章間連携は『クロスモジュール統合分析』章で行う"
  grep -nE "^=+[[:space:]].*(後続分析|後続章|次章への|申し送り|ハンドオフ)" "$REPORT" 2>/dev/null | head -5 | sed 's/^/   該当見出し: /'
  fail=1
else
  echo "✅ Check 8e OK:  工程ナレーション節なし"
fi

# Check 8f: 構造化分析の技法用語（内部用語）の本文露出（WARN・terminology.md「構造化分析の読者向け用語」）
#   ACH・リンチピン・弁別材料・ミラーイメージング等はスキーマ内部の技法名であり、読者向けには
#   「別解釈」「決め手」「見直しのサイン」等の日本語呼称を使う。WARN で導入（既存レポート互換）。
jargon_leak=$(count_matches "\bACH\b|リンチピン|ミラーイメージング|弁別材料|競合仮説分析|非弁別" "$REPORT")
if [ -f reports/report_executive.typ ]; then
  jargon_leak=$((jargon_leak + $(count_matches "\bACH\b|リンチピン|ミラーイメージング|弁別材料|競合仮説分析|非弁別" "reports/report_executive.typ")))
fi
if [ "$jargon_leak" -gt 0 ]; then
  echo "⚠️  Check 8f WARN: 構造化分析の技法用語（ACH/リンチピン/弁別材料/ミラーイメージング等）が本文に ${jargon_leak}行"
  grep -nE "\bACH\b|リンチピン|ミラーイメージング|弁別材料|競合仮説分析|非弁別" "$REPORT" 2>/dev/null | head -4 | sed 's/^/   該当: /'
  echo "   → 読者向け呼称に置換: 対立仮説→「別解釈」、弁別材料→「決め手（この解釈と食い違う事実）」、残った仮説→「最も矛盾の少ない解釈」、リンチピン→「結論の前提と見直しのサイン」（terminology.md「構造化分析の読者向け用語」）"
else
  echo "✅ Check 8f OK:  技法用語の本文露出なし"
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

# Check 11s: 分析の立場 (narrative_stance) の一貫性 (terminology.md §6-2-B)
# Phase A STOP-GATE C で確定した narrative_stance を reports/_phase_a_decisions.json から読み取り、
# 立場が neutral / competitor のとき、対象企業を一人称「当社/弊社」で呼ぶ違反を検出する。
# 母集団タイプとは独立 (単一企業でも投資家・アナリスト視点なら三人称が正)。
echo ""
echo "--- Check 11s: 分析の立場 (誰の意思決定のためか) の一貫性 (terminology.md §6-2-B) ---"
if [ ! -f "$DECISIONS_FILE" ]; then
  echo "⚠️  Check 11s WARN: $DECISIONS_FILE が無いため立場チェックをスキップ"
else
  stance=$(python3 -c "import json; print(json.load(open('$DECISIONS_FILE')).get('narrative_stance',{}).get('code',''))" 2>/dev/null)
  if [ -z "$stance" ]; then
    echo "⚠️  Check 11s WARN: narrative_stance.code が未設定 (旧セッション or Phase A で立場未確定)。Phase A STOP-GATE C で立場 (self/competitor/buyer/supplier/neutral) を確定してください"
  else
    echo "ℹ️  Check 11s: 分析の立場 = ${stance}"
    first_person=$(count_matches "当社|弊社|我が社|わが社" "$REPORT")
    if [ -f reports/report_executive.typ ]; then
      first_person=$((first_person + $(count_matches "当社|弊社|我が社|わが社" "reports/report_executive.typ")))
    fi
    case "$stance" in
      neutral)
        if [ "$first_person" -gt 0 ]; then
          echo "❌ Check 11s FAIL: 立場=neutral(中立/投資家・アナリスト)なのに一人称「当社/弊社」が ${first_person}件。対象企業は企業名・「同社」で三人称にする (terminology.md §6-2-B)"
          fail=1
        else
          echo "✅ Check 11s OK:  立場=neutral で一人称の混入なし (三人称で一貫)"
        fi
        ;;
      competitor|buyer|supplier)
        if [ "$first_person" -gt 0 ]; then
          echo "⚠️  Check 11s WARN: 立場=${stance}(関係性立場) で「当社」等が ${first_person}件。読み手(自社自身)を指すなら可だが、対象企業を指すなら違反。目視確認のこと"
        else
          echo "✅ Check 11s OK:  立場=${stance}(関係性立場) で一人称の混入なし"
        fi
        ;;
      self)
        echo "✅ Check 11s OK:  立場=self (自社視点) のため「当社」使用は適切"
        ;;
      *)
        echo "⚠️  Check 11s WARN: 未知の立場コード '${stance}' (self/competitor/neutral のいずれかであるべき)"
        ;;
    esac
  fi
fi

# Check 11s': 関係性立場 (competitor/buyer/supplier) で自社名 (own_company) がレポート本文に登場するか (WARN・terminology.md §6-2-B / data_notes.md §3)
#   関係性立場では自社を Web 調査し、対象企業との対比に用いるのが本旨 (Phase A STOP-GATE C で own_company を確定)。
#   自社名が本文に一度も出てこない＝自社視点の対比が書かれておらず「競合/買い手/供給側として」の一般論に留まっている疑い。
echo ""
echo "--- Check 11s': 関係性立場での自社名 (own_company) の本文反映 (WARN) ---"
if [ ! -f "$DECISIONS_FILE" ]; then
  echo "ℹ️  Check 11s': $DECISIONS_FILE が無いためスキップ"
else
  stance_code=$(python3 -c "import json; print(json.load(open('$DECISIONS_FILE')).get('narrative_stance',{}).get('code','') or '')" 2>/dev/null)
  own_company=$(python3 -c "import json; print(json.load(open('$DECISIONS_FILE')).get('narrative_stance',{}).get('own_company','') or '')" 2>/dev/null)
  case "$stance_code" in competitor|buyer|supplier) is_relational=1 ;; *) is_relational=0 ;; esac
  if [ "$is_relational" -eq 0 ]; then
    echo "ℹ️  Check 11s': 立場=${stance_code:-未設定} (関係性立場でない) のため自社対比チェックは対象外"
  elif [ -z "$own_company" ]; then
    echo "ℹ️  Check 11s': 立場=${stance_code} だが own_company 未設定 (自社名を伏せる選択)。一般的な視点として扱いスキップ"
  else
    own_hits=$(grep -F -c "$own_company" "$REPORT" 2>/dev/null | head -n1 | tr -d ' \n')
    own_hits=${own_hits:-0}
    if [ -f reports/report_executive.typ ]; then
      exec_hits=$(grep -F -c "$own_company" "reports/report_executive.typ" 2>/dev/null | head -n1 | tr -d ' \n')
      own_hits=$((own_hits + ${exec_hits:-0}))
    fi
    if [ "$own_hits" -eq 0 ]; then
      echo "⚠️  Check 11s' WARN: 立場=${stance_code}(関係性立場) で自社「${own_company}」がレポート本文に 0 件。自社を Web 調査し、対象企業との対比 (自社の強み/弱み/空白 vs 対象企業) を最低1章で書くこと (terminology.md §6-2-B、data_notes.md §3「自社の調査」)"
    else
      echo "✅ Check 11s' OK:  自社「${own_company}」がレポートに ${own_hits} 件登場 (対象企業との対比あり)"
    fi
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

# Check 14: メタラベル（4層モデルのラベル）の本文露出検出 (common_framework.md §1)
# 「（事実）」「（解釈）」「（So what）」「経営的含意:」等を本文に書くのは禁止。4層は思考の枠組みであり、
# 地の文に溶け込ませる。ラベルを書くと機械生成感が出て読み物として不合格。
echo ""
echo "--- Check 14: メタラベルの本文露出 (（事実）/（解釈）/（So what）/経営的含意: 等) ---"
label_leak=$(count_matches "（事実）|（解釈）|（提言）|（事実・|（予測）|（So ?[Ww]hat|So ?[Ww]hat[:：]|経営的含意[:：]|経営的意味[:：]|経営的示唆[:：]|（L[1-4]|【事実】|【解釈】" "$REPORT")
if [ "$label_leak" -gt 0 ]; then
  echo "❌ Check 14 FAIL: 4層モデルのラベルが本文に ${label_leak} 行で露出しています"
  echo "   → 「（事実）」「（解釈）」「（So what）」「（提言）」「経営的含意:」「（L1）」等のラベルを本文に書くのは禁止"
  echo "   → 4層は思考の枠組み。地の文に自然に溶け込ませる（接続表現で1つの流れる文章に）"
  echo "   → 詳細: analysis/common_framework.md §1（最重要: 4層は本文に書くラベルではない）"
  fail=1
else
  echo "✅ Check 14 OK:  4層モデルのラベル露出なし"
fi

# Check 15: 国籍トートロジー検出 (terminology.md §6)
# 母集団が国・地域で絞られている場合（例: 日本出願）、その国が「牽引/主導/リード」は母集団定義上当然で
# 技術的優位の根拠にならない（循環論法）。検出は WARN（母集団が国別でない場合は正当なこともあるため）。
echo ""
echo "--- Check 15: 国籍トートロジー（母集団国が『牽引/主導/リード』）---"
nat_lead=$(count_matches "(日本|国内|日本企業|日本勢|我が国|邦人)(企業|勢)?が[^。]{0,24}(牽引|主導|リード|世界を)" "$REPORT")
if [ "$nat_lead" -gt 0 ]; then
  echo "⚠️  Check 15 WARN: 母集団国（日本等）が『牽引/主導/リード』と書かれた箇所が ${nat_lead} 行あります"
  echo "   → 母集団が国・地域で絞られている場合（例: 日本出願）、その国の出願が多い/主導は母集団定義上当然で、"
  echo "     技術的優位の根拠になりません（循環論法）。削除するか限界注記に置換してください。"
  echo "   → ❌「本母集団では日本企業が成長を主導している」"
  echo "   → ✅「本母集団は日本語公報に限定されるため、海外勢の出願動向は本分析の評価対象外である」"
  echo "   → 詳細: analysis/terminology.md §6（国籍・地域のスコープ）"
fi

# Check 17: 過剰修辞・気取った比喩の検出（レポート本文・WARN・terminology.md §6.5）
echo ""
echo "--- Check 17: 過剰修辞・気取った比喩（平易・正確に） ---"
rhetoric=$(count_matches "越境者|狼煙|地殻変動|試金石|旗手|号砲|幕開け|夜明け|胎動|風雲児|金字塔|羅針盤|苗床|三段ロケット|主役交代" "$REPORT")
if [ "$rhetoric" -gt 0 ]; then
  echo "⚠️  Check 17 WARN: 過剰修辞・気取った比喩が ${rhetoric} 行（越境者/狼煙/地殻変動/旗手 等）"
  echo "   → 比喩を避け、その比喩が指す事実・数値を平易に書く（例「越境者」→「領域横断的な出願人」）"
  echo "   → 詳細: analysis/terminology.md §6.5（記述スタイル）"
else
  echo "✅ Check 17 OK:  過剰修辞・気取った比喩なし"
fi

# Check 18: 経営層向け要約版（別冊）の充実度（存在時のみ・executive_summary_guide.md は 8-12 ページ）
echo ""
echo "--- Check 18: 別冊（report_executive.typ）の充実度 ---"
EXEC="reports/report_executive.typ"
if [ ! -f "$EXEC" ]; then
  echo "ℹ️  Check 18: 別冊未生成のためスキップ（別冊生成フラグ OFF なら正常）"
else
  exec_lines=$(wc -l < "$EXEC" | tr -d ' ')
  exec_notes=$(count_matches 'note-box|callout|#rect|#block' "$EXEC")
  if [ "$exec_lines" -lt 120 ]; then
    echo "❌ Check 18 FAIL: 別冊 = ${exec_lines}行（要 120行以上）。表紙＋数段落だけの薄い別冊は不合格"
    echo "   → executive_summary_guide.md に従い 8-12 ページ（発見ごとに 8-12 行＋数値 callout・ページ充填）に拡充する"
    fail=1
  else
    echo "✅ Check 18 OK:   別冊 = ${exec_lines}行（120行以上）・callout/note-box=${exec_notes}"
  fi
  # Check 18b: 別冊の「結論の検証・見直しのサイン」（WARN・executive_summary_guide.md）
  #   経営層向けには「なぜこの結論を信じてよいか（別解釈の検討）」「何が起きたら見直すか
  #   （監視トリガー）」が最重要。別冊にこの要素が無ければ WARN。
  exec_ach=$(count_matches "見直しのサイン|見直すべきサイン|別解釈|決め手" "$EXEC")
  if [ "$exec_ach" -eq 0 ]; then
    echo "⚠️  Check 18b WARN: 別冊に「別解釈の検討」「見直しのサイン」の記述が無い（経営層が最も必要とする要素）"
    echo "   → 主要な発見ごとに「検討した別解釈と決め手」を1〜2行、末尾に「この結論を見直すべきサイン」ボックスを置く（executive_summary_guide.md）"
  else
    echo "✅ Check 18b OK:  別冊に検証・見直しサインの記述 ${exec_ach}行"
  fi
fi

# Check 19: 反復・水増し検出（report.typ / report_executive.typ・FAIL）
# literal なモデル（Codex 等）が「最低N行」を同一文のコピペループで充足する水増しを機械検出する。
# 合否は「量(行数)」ではなく「内容の固有性(非重複)」で決める。行数ゲート(Check 1)を素通りした
# 水増しレポートはこの Check で落とす。slides_spec の PPTX 重複検出(16h/16j)を本文側に展開したもの。
echo ""
echo "--- Check 19: 反復・水増し検出 (同一文の反復/回転名詞テンプレ/連番定型見出し/接続句ローテ/本文スクリプト生成) ---"
# Check 19a: レポート本文を Python スクリプトで機械生成していないか（水増しの温床・FAIL）
# reports/ 内の .py が .typ を書き出している＝本文をテンプレート生成している強い証拠。
# （PPTX 生成スクリプトは .pptx を書くので .typ では誤検出しない）
gen_scripts=$(find reports -maxdepth 1 -name '*.py' 2>/dev/null | while read -r f; do
  if grep -qE "\.typ" "$f" 2>/dev/null; then echo "$f"; fi
done)
if [ -n "$gen_scripts" ]; then
  echo "❌ Check 19a FAIL: レポート本文を生成する Python スクリプトを検出（reports/ 内の .py が .typ を書き出している）"
  echo "$gen_scripts" | sed 's/^/     /'
  echo "   → レポート本文・deep_dive.typ をスクリプトでテンプレート生成するのは禁止。各文を固有の分析として直接書く"
  echo "   → 生成スクリプトを削除し本文を手で書き直すこと（SKILL.md §0・analysis/deep_dive_guide.md「記述品質の絶対基準」）"
  fail=1
else
  echo "✅ Check 19a OK:  本文のスクリプト機械生成なし"
fi
python3 - "$REPORT" "reports/report_executive.typ" <<'PYEOF'
import sys, os, re
from collections import Counter

# 本文ではない Typst のコード/レイアウト行を除外する接頭辞
SKIP = re.compile(r'^#(import|include|show|set|let|pagebreak|colbreak|figure|image|'
                  r'grid|table|place|line|rect|block|stack|columns|page|v\(|h\()')
# 構造化カード等の名前付き引数行（key: "値" / key: 数値）は本文ではない（誤検出回避）
ARG_SKIP = re.compile(r'^[A-Za-z_][A-Za-z0-9_\-]*\s*:\s*("[^"]*"|[0-9０-９.\-]+)\s*,?\s*$')

def load_sentences(path):
    try:
        raw = open(path, encoding='utf-8').read()
    except Exception:
        return None, None
    keep = []
    for ln in raw.split('\n'):
        s = ln.strip()
        if not s or s.startswith('//') or SKIP.match(s) or ARG_SKIP.match(s):
            continue
        keep.append(ln)
    body = '\n'.join(keep)
    parts = re.split(r'[。\n]', body)   # 文分割: 句点と改行
    return parts, raw

def norm(s):
    # 関数名(#evidence-box 等)と参照・装飾記号・括弧・引用符を除去し、内部の地の文だけ残す
    t = re.sub(r'#[a-zA-Z][a-zA-Z0-9\-]*', '', s)   # 関数名のみ削除（中身は残す）
    t = re.sub(r'@[a-zA-Z0-9_\-]+', '', t)
    t = re.sub(r'[\[\]*_`#<>|=()（）"\'「」『』｛｝{}]', '', t)
    t = re.sub(r'\s+', '', t)
    return t.strip('　 。．、,.;:！？!?・-—~〜')

def tnorm(s):
    # 回転名詞（カタカナ語・英数）を 〇 に潰し、定型テンプレートの量産を検出可能にする
    t = re.sub(r'[ァ-ヴ][ァ-ヴー・]{1,}', '〇', s)
    t = re.sub(r'[A-Za-z][A-Za-z0-9./\-]*', '〇', t)
    t = re.sub(r'[0-9０-９,]+', '', t)
    return t

def analyze(path, exact_min, tmpl_min, ratio_max, enum_min, tail_min, label):
    parts, raw = load_sentences(path)
    if parts is None:
        return 0
    sents = [norm(p) for p in parts]
    sents = [s for s in sents if len(s) >= 20]
    total = len(sents)
    if total == 0:
        print("ℹ️  Check 19 [%s]: 判定対象の地の文が抽出できません（スキップ）" % label)
        return 0
    c = Counter(sents)
    exact_off = sorted([(s, v) for s, v in c.items() if v >= exact_min], key=lambda x: -x[1])
    tc = Counter([tnorm(s) for s in sents if len(s) >= 24])
    tmpl_off = sorted([(s, v) for s, v in tc.items() if v >= tmpl_min and len(s) >= 24], key=lambda x: -x[1])
    # 接続句ローテーション対策: 文末22字で重複を測る（先頭の接続詞だけ変えても文末は同一）
    tails = Counter(s[-22:] for s in sents if len(s) >= 24)
    tail_off = sorted([(t, v) for t, v in tails.items() if v >= tail_min], key=lambda x: -x[1])
    dup_sents = sum(v for s, v in c.items() if v >= 2)
    ratio = dup_sents / total
    enum = len(re.findall(r'(観点|論点|考察|事例|ポイント|要点|事項|示唆)\s*[0-9０-９]{1,3}', raw))

    print("ℹ️  Check 19 [%s]: 地の文=%d文, 重複文比率=%.0f%%, 連番定型見出し=%d件" % (label, total, ratio * 100, enum))
    fail = 0
    if exact_off:
        fail = 1
        print("❌ Check 19 FAIL [%s]: 同一文の反復が %d 種（各 %d 回以上）。コピペ水増し＝自動不合格" % (label, len(exact_off), exact_min))
        for s, v in exact_off[:6]:
            print("     %d回: %s" % (v, s[:50]))
    if tmpl_off:
        fail = 1
        print("❌ Check 19 FAIL [%s]: 回転名詞だけ変えた定型文の量産が %d 種（各 %d 回以上）" % (label, len(tmpl_off), tmpl_min))
        for s, v in tmpl_off[:4]:
            print("     %d回: %s" % (v, s[:50]))
    if tail_off:
        fail = 1
        print("❌ Check 19 FAIL [%s]: 接続句だけ変えた水増し（同一文末が %d 種・各 %d 回以上）。文体を変えても内容の重複は不合格" % (label, len(tail_off), tail_min))
        for t, v in tail_off[:4]:
            print("     %d回: …%s" % (v, t))
    if ratio > ratio_max:
        fail = 1
        print("❌ Check 19 FAIL [%s]: 重複文比率 %.0f%% > %.0f%%（本文の多くが使い回し）" % (label, ratio * 100, ratio_max * 100))
    if enum >= enum_min:
        fail = 1
        print("❌ Check 19 FAIL [%s]: 「○○観点 1,2,3…」式の連番定型見出しが %d 件（≥%d）。連番で定型文を量産しない" % (label, enum, enum_min))
    if not fail:
        print("✅ Check 19 OK  [%s]: 反復・水増しなし" % label)
    return fail

report = sys.argv[1]
execpath = sys.argv[2] if len(sys.argv) > 2 else None
f1 = analyze(report, exact_min=4, tmpl_min=8, ratio_max=0.15, enum_min=12, tail_min=5, label="本編")
f2 = analyze(execpath, exact_min=3, tmpl_min=6, ratio_max=0.18, enum_min=8, tail_min=4, label="別冊") if (execpath and os.path.isfile(execpath)) else 0
if f1 or f2:
    print("   → 対処: 文を繰り返す代わりに、新しい代表特許(固有の公開番号)・新しい数値根拠・別のクロスパターン・Web裏付けを足す")
    print("   → 各段落は前段落と異なる固有の事実を最低1つ含めること（analysis/deep_dive_guide.md「記述品質の絶対基準」）")
    sys.exit(1)
sys.exit(0)
PYEOF
rep19_rc=$?
if [ "$rep19_rc" -ne 0 ]; then
  fail=1
fi

# Check 20: スナップショット網羅（撮った全マップを分析・掲載する・WARN）
# snapshots/ にあるPNGのうち report.typ で参照されていないものを警告する。
# 「撮ったマップは全て掲載・分析する」という原則（analysis/map_reading.md）の機械チェック。
echo ""
echo "--- Check 20: スナップショット網羅（撮った全マップを掲載・分析）---"
if [ -d snapshots ]; then
  # VOYAGER evidence 画像（voyager_ev*）は CAPCOM 本来マップ（atlas_*/saturn_*/core_* 等）と内容が
  # 重複するため網羅対象から除外する。これらを掲載すると同一マップが二重に出る（マップ重複の主因）。
  # ※ v9 以降は capcom 側で voyager_ev の出力を停止済み（新規 ZIP には含まれない）。
  #    この除外は旧セッションの ZIP との互換のために残置している。
  total_snaps=0; referenced=0; missing=""
  for img in snapshots/*.png; do
    [ -e "$img" ] || continue
    base=$(basename "$img")
    case "$base" in voyager_ev*) continue;; esac
    total_snaps=$((total_snaps + 1))
    if grep -qF "$base" "$REPORT" 2>/dev/null; then
      referenced=$((referenced + 1))
    else
      missing="${missing} ${base}"
    fi
  done
  if [ "$total_snaps" -eq 0 ]; then
    echo "ℹ️  Check 20: snapshots/ に対象PNG（voyager_ev* 以外）がありません（スキップ）"
  elif [ -n "$missing" ]; then
    echo "⚠️  Check 20 WARN: CAPCOM マップ ${total_snaps}枚中 ${referenced}枚のみ report.typ で参照。未掲載:${missing}"
    echo "   → 撮った全マップを分析・掲載するのが原則。意図的除外でなければ、該当章の本文の近く（そのマップを論じる段落）に #snapshot-figure で追加し各マップを分析する"
    echo "   → 掲載だけでは不可＝1枚ごとに読み取り本文を書く（ドリルダウンは「どのクラスタを拡大したか・何が見えるか・代表特許」を必ず添える。分析本文の無い図は Check 29 で検出）"
    echo "   → 各マップの見方は analysis/map_reading.md を参照（種別ヒント一覧に無い名前でも接頭辞→画像内容から種別を判定し、必ず掲載する）"
  else
    echo "✅ Check 20 OK:  CAPCOM マップ全 ${total_snaps}枚が report.typ で参照されている"
  fi
  # Check 20b: VOYAGER evidence（voyager_ev*）の掲載検出（本来マップと重複するので使わない・WARN）
  # ※ v9 以降は新規 ZIP に voyager_ev が含まれないため通常 0 件。旧 ZIP 互換で検出を残置。
  voyager_used=$(grep -oE 'voyager_ev[0-9_]+\.png' "$REPORT" 2>/dev/null | sort -u | wc -l | tr -d ' ')
  if [ "$voyager_used" -gt 0 ]; then
    echo "⚠️  Check 20b WARN: VOYAGER evidence 画像（voyager_ev*）が ${voyager_used}種 掲載されている。これらは CAPCOM 本来マップ（atlas_*/saturn_*/core_* 等）と内容が重複するので掲載しない（マップ重複の主因）。CAPCOM のモジュール別スナップショットを使うこと"
  fi
else
  echo "ℹ️  Check 20: snapshots/ ディレクトリなし（スキップ）"
fi

# Check 21: 代表特許番号のプレースホルダ／捏造検出（FAIL・patent_citation.md §5）
# 「特開2023-XXXXXX」のような XXXX プレースホルダや、出願年から推測した番号は禁止。
# 公開番号は必ず data/patents.csv の実番号を引く（未公開時は「出願番号 XXXX-XXXXXX」を実番号で）。
echo ""
echo "--- Check 21: 特許番号のプレースホルダ／捏造検出 ---"
placeholder_pat=$(count_matches "(特開|特願|特表|特許第|公開番号|出願番号)[^0-9]{0,6}X{3,}|[0-9]{4}-X{3,}|X{4,}" "$REPORT")
if [ "$placeholder_pat" -gt 0 ]; then
  echo "❌ Check 21 FAIL: 特許番号にプレースホルダ(XXXX)／捏造番号が ${placeholder_pat}件 残存"
  grep -nE "(特開|特願|特表|特許第|公開番号|出願番号)[^0-9]{0,6}X{3,}|[0-9]{4}-X{3,}|X{4,}" "$REPORT" 2>/dev/null | head -6 | sed 's/^/   該当: /'
  echo "   → 公開番号は必ず data/patents.csv から実番号を引く（発明名称＋applicant_main＋year でフィルタ）"
  echo "   → 未公開（公開番号が空）の場合は出願番号を『出願番号 2023-192478』のように実番号で記載する"
  echo "   → 出願年から推測した番号や XXXXXX プレースホルダは禁止（詳細: analysis/patent_citation.md §5）"
  fail=1
else
  echo "✅ Check 21 OK:  特許番号のプレースホルダ／捏造なし"
fi

# Check 22: 見出し・目次の数値強調漏れ予防（WARN・report_style.typ / report_structure.md）
# 見出しに「数値＋単位」を入れると本文用の自動強調(navy太字)が漏れて色が不揃いになりやすい。
# テンプレ側(no-num-hl の U+2060 方式)で解消済みだが、取りこぼし検知と可読性のため WARN する。
# また no-num-hl で包まれていない裸の #outline も検出する（目次の数値強調漏れの原因）。
echo ""
echo "--- Check 22: 見出しの数値+単位／裸の #outline（数値強調漏れ予防） ---"
heading_num=$(count_matches "^=+ .*[0-9]+ *(件|社|%|％|倍|ポイント)" "$REPORT")
bare_outline=$(count_matches "^#outline\(" "$REPORT")
if [ "$heading_num" -gt 0 ]; then
  echo "⚠️  Check 22 WARN: 見出しに『数値＋単位』が ${heading_num}件（自動強調が漏れて色が不揃いになりやすい）"
  grep -nE "^=+ .*[0-9]+ *(件|社|%|％|倍|ポイント)" "$REPORT" 2>/dev/null | head -5 | sed 's/^/   該当見出し: /'
  echo "   → 見出しは定性表現にし、数値はサブメッセージ・要点・本文に置く（例: 上位5社→主要出願人、30件以上→大規模クラスタ）"
fi
if [ "$bare_outline" -gt 0 ]; then
  echo "⚠️  Check 22 WARN: no-num-hl で包まれていない裸の #outline が ${bare_outline}件（目次の数値強調が漏れる）"
  echo "   → #no-num-hl(outline(...)) のように no-num-hl で包む（詳細: analysis/report_structure.md §1）"
fi
if [ "$heading_num" -eq 0 ] && [ "$bare_outline" -eq 0 ]; then
  echo "✅ Check 22 OK:  見出しの数値+単位・裸の #outline なし"
fi

# Check 23: マップの章頭羅列検出（WARN・common_framework.md §4 / report_structure.md）
#   見出し（=, ==, ===）の直後に本文を書かず #snapshot-figure を置くアンチパターンを検出。
#   マップは「論じる本文段落の近くに1枚ずつ」が原則。見出し直後の図は飾りになりやすい。
echo ""
echo "--- Check 23: マップの章頭羅列（見出し直後の #snapshot-figure・WARN） ---"
head_fig_lines=$(awk '
  /^=+[[:space:]]/ { pend=1; next }
  pend==1 {
    if ($0 ~ /^[[:space:]]*$/) { next }      # 見出しと本文の間の空行はスキップ
    if ($0 ~ /#snapshot-figure/) { printf "   L%d: %s\n", NR, $0 }
    pend=0                                     # 最初の非空行を見たら判定終了
  }
' "$REPORT" 2>/dev/null)
if [ -n "$head_fig_lines" ]; then
  head_fig_count=$(printf '%s\n' "$head_fig_lines" | grep -c .)
  echo "⚠️  Check 23 WARN: 見出し直後に #snapshot-figure を配置した章が ${head_fig_count}件（マップは見出し直後に羅列せず、論じる本文段落の近くに1枚ずつ置く）"
  printf '%s\n' "$head_fig_lines" | head -8
  echo "   → 各図の直前に、その図から読み取った事実→解釈の本文段落を書く（common_framework.md §4「マップ（スナップショット画像）の配置ルール」）"
else
  echo "✅ Check 23 OK:  見出し直後へのマップ羅列なし"
fi

# Check 24: Explorer 章の「象限」誤用検出（FAIL・map_reading.md §2 / explorer_exemplar.typ §4）
#   APOLLO の Explorer に「成長率×中心性の象限図（散布図）」は存在しない。急成長キーワードの評価は
#   図を持たないテキスト分析であり「象限」「4象限マトリクス」「第N象限」と呼んではならない（捏造の温床）。
#   除外: ①中心性を伴わない、かつ MEGA(CAGR×活動量4象限)・ATLAS(出願数×権利化率の象限)・
#         クラスタ動態マップ(累積件数×CAGR)への正当なクロス参照行。
#   「中心性」と「象限」が同一行に共起したら、いかなるキーワードがあっても捏造としてFAIL。
echo ""
echo "--- Check 24: Explorer 章の「象限」誤用（FAIL・成長率×中心性の象限図は存在しない） ---"
exp_quad_lines=$(awk '
  /^=[[:space:]]/ {
    if ($0 ~ /Explorer/ || $0 ~ /共起ネットワーク/) { in_exp=1 } else { in_exp=0 }
    next
  }
  in_exp==1 && /象限/ {
    if ($0 ~ /中心性/) { printf "   L%d: %s\n", NR, $0; next }   # 中心性×象限 = 必ず捏造
    # MEGA / ATLAS権利化率 / クラスタ動態 への正当なクロス参照は除外
    if ($0 ~ /MEGA|活動量|CAGR|ATLAS|権利化率|出願数|クラスタ動態|累積件数|Saturn|EAGLE|NEBULA/) { next }
    printf "   L%d: %s\n", NR, $0
  }
' "$REPORT" 2>/dev/null)
if [ -n "$exp_quad_lines" ]; then
  exp_quad_count=$(printf '%s\n' "$exp_quad_lines" | grep -c .)
  echo "❌ Check 24 FAIL: Explorer 章に不正な「象限」が ${exp_quad_count}件（APOLLO に『成長率×中心性の象限図』は存在しない。MEGA/ATLAS権利化率/クラスタ動態への正当な参照は除外済み）"
  printf '%s\n' "$exp_quad_lines" | head -8
  echo "   → exemplars/explorer_exemplar.typ §4「急成長キーワードの戦略的評価」に統一。「象限」「4象限マトリクス」「第N象限」を使わない（analysis/map_reading.md §2）"
  fail=1
else
  echo "✅ Check 24 OK:  Explorer 章に象限の誤用なし"
fi

# Check 25: 章末まとめ（本章のまとめ）の有無（WARN・report_structure.md / deep_dive_guide.md）
#   各モジュール分析章（= 見出し単位）の末尾に #chapter-summary[...] を置く運用。
#   走査層の読者が章を精読せずとも結論を掴めるようにする締め要素。任意WARN（無くてもFAILしない）。
#   対象は分析章のみ（Executive Summary・結論・付録・追加的事項章は対象外）。
echo ""
echo "--- Check 25: 章末まとめ（📌 本章のまとめ）の有無（WARN） ---"
missing_summary=$(awk '
  function flush() {
    if (in_mod==1 && has_sum==0) { printf "   章「%s」に #chapter-summary なし\n", htext }
  }
  /^=[[:space:]]/ {
    flush()
    htext = $0; sub(/^=[[:space:]]+/, "", htext)
    if ($0 ~ /ATLAS|CORE|Saturn|MEGA|Explorer|CREW|EAGLE|NEBULA|外部環境分析|共起ネットワーク|俯瞰図|出願人動態|基本統計|技術課題|ネットワーク分析/) { in_mod=1 } else { in_mod=0 }
    has_sum=0
    next
  }
  /#chapter-summary/ { has_sum=1 }
  END { flush() }
' "$REPORT" 2>/dev/null)
total_summary=$(count_matches "#chapter-summary" "$REPORT")
if [ -n "$missing_summary" ]; then
  missing_summary_count=$(printf '%s\n' "$missing_summary" | grep -c .)
  echo "⚠️  Check 25 WARN: #chapter-summary（本章のまとめ）が無い分析章が ${missing_summary_count}件（章末まとめは任意だが全章で推奨。現在の総数=${total_summary}個）"
  printf '%s\n' "$missing_summary" | head -8
  echo "   → 各分析章の末尾に #chapter-summary[...] を置き、本章の要点を3〜5行で凝縮する（deep_dive_guide.md「読みやすさ（走査層）」④）"
else
  echo "✅ Check 25 OK:  全分析章に #chapter-summary あり（総数=${total_summary}個）"
fi

# Check 26: 章末まとめの位置（#chapter-summary の後に本文が続いていないか・WARN）
#   ルール: 各章は本文（段落・マップ・evidence）を全て書き終えてから、最後に #chapter-summary を
#   1つ置いて章を閉じる（deep_dive_guide.md「読みやすさ（走査層）」④ / report_structure.md 章共通ルール）。
#   まとめを書いた後に本文を足すと、まとめが章の途中に取り残される（執筆順ルールの backstop）。
#   判定: 章内で #chapter-summary ブロック（[ ] の括弧深度で終端判定）終了後、
#   次の = 見出しまでに実質行（非空・非コメント・非レイアウト）があれば WARN。
echo ""
echo "--- Check 26: 章末まとめの位置（まとめの後に本文が続く章・WARN） ---"
sum_pos_lines=$(awk '
  function report() {
    if (seen_sum==1 && in_sum==0 && tail>0) {
      printf "   章「%s」: #chapter-summary の後に本文が %d 行\n", htext, tail
    }
  }
  /^=[[:space:]]/ {
    report()
    htext=$0; sub(/^=+[[:space:]]+/, "", htext)
    seen_sum=0; in_sum=0; depth=0; tail=0
    next
  }
  {
    if (in_sum==1) {
      depth += gsub(/\[/, "[") - gsub(/\]/, "]")
      if (depth <= 0) { in_sum=0 }
      next
    }
    if ($0 ~ /#chapter-summary/) {
      seen_sum=1; tail=0
      rest=$0
      sub(/^.*#chapter-summary/, "", rest)
      d = gsub(/\[/, "[", rest) - gsub(/\]/, "]", rest)
      if (d > 0) { in_sum=1; depth=d }
      next
    }
    if ($0 ~ /^[[:space:]]*$/) { next }
    if ($0 ~ /^[[:space:]]*\/\//) { next }
    if ($0 ~ /^[[:space:]]*#(pagebreak|colbreak|v\(|h\()/) { next }
    if (seen_sum==1) { tail++ }
  }
  END { report() }
' "$REPORT" 2>/dev/null)
if [ -n "$sum_pos_lines" ]; then
  sum_pos_count=$(printf '%s\n' "$sum_pos_lines" | grep -c .)
  echo "⚠️  Check 26 WARN: #chapter-summary が章末に無い（まとめの後に本文が続く）章が ${sum_pos_count}件"
  printf '%s\n' "$sum_pos_lines" | head -8
  echo "   → 執筆順ルール: 章の本文を全て書き終えてから、最後に #chapter-summary を1つ追記して章を閉じる。まとめの後に本文・図を足さない"
  echo "   → 追記が必要になった場合は、まとめを一旦削除 → 本文を追加 → まとめを書き直して再び章末に置く（deep_dive_guide.md「読みやすさ（走査層）」④）"
else
  echo "✅ Check 26 OK:  #chapter-summary の後に本文が続く章なし"
fi

# Check 27: ワードクラウド分析の痕跡（頻出語＋出現回数の本文列挙・WARN）
#   snapshots/ にワードクラウド画像があるのに、本文に「ワードクラウド」への言及が無い、
#   または『語（N回）』形式の出現回数表記が少ない場合、「貼って終わり」の疑い（map_reading.md §6）。
echo ""
echo "--- Check 27: ワードクラウド分析（頻出語＋出現回数の本文列挙・WARN） ---"
wc_snaps=$(ls snapshots/*wordcloud* 2>/dev/null | wc -l | tr -d ' ')
if [ "$wc_snaps" -eq 0 ]; then
  echo "ℹ️  Check 27: ワードクラウドのスナップショットなし（スキップ）"
else
  wc_mentions=$(count_matches "ワードクラウド" "$REPORT")
  freq_notation=$(grep -oE "（[0-9,，]+回）|\([0-9,]+回\)" "$REPORT" 2>/dev/null | wc -l | tr -d ' ')
  freq_notation=${freq_notation:-0}
  if [ "$wc_mentions" -eq 0 ] || [ "$freq_notation" -lt 5 ]; then
    echo "⚠️  Check 27 WARN: ワードクラウド画像 ${wc_snaps}枚 に対し、本文の言及=${wc_mentions}行・『語（N回）』形式の出現回数表記=${freq_notation}件（要 5件以上）"
    echo "   → ワードクラウドは貼るだけでは不可。上位頻出語を『水電解』（87回）のように語＋出現回数で最低5語列挙し、技術的含意（何の技術か・どんな課題か・なぜ高頻度か）まで論じる"
    echo "   → 出現回数は data/*_wordcloud.json の word_frequencies の実数を引く（推測で書かない）。読み方: analysis/map_reading.md §6 Step 1-5"
  else
    echo "✅ Check 27 OK:  ワードクラウド言及=${wc_mentions}行・出現回数表記=${freq_notation}件"
  fi
fi

# Check 28: マップの横並べ・縮小（grid 2枚並置 / width<100%・WARN）
#   マップは1枚1行・全幅が原則（common_framework.md §4）。grid(columns: 2, ...) に2枚の
#   マップを入れると各マップが縮小されて読めなくなる。snapshots/ 画像の width<100% 縮小も検出。
echo ""
echo "--- Check 28: マップの横並べ・縮小（grid 2枚並置 / width<100%・WARN） ---"
python3 - "$REPORT" <<'PYEOF'
import sys, re
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
n = len(lines)
viol_grid = []   # (行番号, ブロック内の画像数)
viol_width = []  # (行番号, width%)
i = 0
while i < n:
    ln = lines[i]
    m = re.search(r'\bgrid\(', ln)
    if m:
        depth = 0
        imgs = 0
        start = i
        j = i
        while j < n and j < start + 80:
            s = ln[m.start():] if j == i else lines[j]
            depth += s.count('(') - s.count(')')
            imgs += len(re.findall(r'snapshot-figure|image\(', s))
            j += 1
            if depth <= 0:
                break
        if imgs >= 2:
            viol_grid.append((start + 1, imgs))
        i = j
        continue
    mw = re.search(r'image\([^)]*width:\s*([0-9]{1,2})\s*%', ln)
    if mw and 'snapshots/' in ln:
        viol_width.append((i + 1, mw.group(1)))
    i += 1
if viol_grid or viol_width:
    if viol_grid:
        print("⚠️  Check 28 WARN: grid ブロック内にマップ/画像が2枚以上入っている箇所が %d 件（横並べ＝縮小で読めなくなる）" % len(viol_grid))
        for lno, cnt in viol_grid[:6]:
            print("   L%d: grid 内に画像 %d 枚" % (lno, cnt))
    if viol_width:
        print("⚠️  Check 28 WARN: snapshots/ 画像を width<100%% で縮小している箇所が %d 件" % len(viol_width))
        for lno, w in viol_width[:6]:
            print("   L%d: width: %s%%" % (lno, w))
    print("   → マップは1枚1行・全幅（#snapshot-figure は width 100%）。並べて比較したい場合も別々の段落に分けて縦に置き、本文で比較を論じる（common_framework.md §4）")
else:
    print("✅ Check 28 OK:  マップの横並べ・縮小なし")
PYEOF

# Check 29: 分析本文が隣接しない図（#snapshot-figure の孤立・WARN）
#   Check 20 は「掲載」しか見ないため、貼るだけで分析テキストが無い図を検出できない。
#   各 #snapshot-figure の直前・直後（空行スキップ）のどちらにも本文段落が無い
#   （どちらも見出し / 別の図 / 章末まとめ / ファイル末尾）場合を「分析されていない図」として WARN。
echo ""
echo "--- Check 29: 分析本文が隣接しない図（貼るだけのマップ・WARN） ---"
python3 - "$REPORT" <<'PYEOF'
import sys, re
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
n = len(lines)

def kind(s):
    t = s.strip()
    if t == '' or t.startswith('//'):
        return 'blank'
    if re.match(r'^=+\s', t):
        return 'heading'
    if '#snapshot-figure' in t or re.match(r'^#(figure|image)\b', t) or re.match(r'^caption\s*:', t) or t in (')', '),'):
        return 'figure'
    if t.startswith('#chapter-summary'):
        return 'summary'
    if re.match(r'^#(pagebreak|colbreak|v\(|h\()', t):
        return 'blank'
    return 'body'

def img_name(ln):
    m = re.search(r'snapshots/([^"\')]+)', ln)
    return m.group(1) if m else '(画像名不明)'

def para_text(idx, direction):
    """図の直前(direction=-1)/直後(+1)の連続する本文段落のテキストを返す"""
    out, j = [], idx + direction
    while 0 <= j < n and kind(lines[j]) == 'blank' and lines[j].strip() == '':
        j += direction
    while 0 <= j < n and kind(lines[j]) == 'body':
        out.append(lines[j])
        j += direction
    return ' '.join(out)

# 図を名指しする指示語（「俯瞰図（下図）を見ると」「図から読み取れる」の型）
DEICTIC = re.compile(r'下図|上図|次図|次の図|以下の図|上記の図|下記の図|図中|本図|この図|左図|右図'
                     r'|に示す|を以下に示|を次に示|図を見る|図を見ると|図から|図が示す|図では|マップでは|マップを見る|マップから')
PATNUM = re.compile(r'[0-9]{4}-[0-9]{5,7}')

viol = []        # 29: 本文段落が隣接しない
viol_ref = []    # 29b: 隣接段落が図を名指ししていない
viol_drill = []  # 29c: ドリルダウンの分析が薄い
for i, ln in enumerate(lines):
    if '#snapshot-figure' not in ln:
        continue
    prev = 'end'
    for j in range(i - 1, -1, -1):
        k = kind(lines[j])
        if k != 'blank':
            prev = k
            break
    nxt = 'end'
    for j in range(i + 1, n):
        k = kind(lines[j])
        if k != 'blank':
            nxt = k
            break
    img = img_name(ln)
    if prev in ('heading', 'figure', 'end') and nxt in ('heading', 'figure', 'summary', 'end'):
        viol.append((i + 1, img))
    else:
        # 29b: 隣接段落はあるが、図を名指し（下図/図中 等）していない＝図と無関係な本文の疑い
        near = para_text(i, -1) + ' ' + para_text(i, +1)
        if not DEICTIC.search(near):
            viol_ref.append((i + 1, img))
    # 29c: ドリルダウン図は近傍±15行に代表特許番号と十分な本文量が必要
    if 'drill' in img.lower():
        lo, hi = max(0, i - 15), min(n, i + 16)
        wtext = '\n'.join(l for l in lines[lo:hi] if kind(l) == 'body')
        wchars = len(re.sub(r'\s', '', wtext))
        has_pat = bool(PATNUM.search(wtext))
        if (not has_pat) or wchars < 300:
            viol_drill.append((i + 1, img, wchars, has_pat))
if viol:
    print("⚠️  Check 29 WARN: 直前・直後のどちらにも本文段落が無い #snapshot-figure が %d 枚（貼るだけ＝分析されていない疑い）" % len(viol))
    for lno, img in viol[:8]:
        print("   L%d: %s" % (lno, img))
    print("   → 各図の直前または直後に、その図から読み取った事実→解釈の本文段落を必ず書く（common_framework.md §4 配置ルール）")
else:
    print("✅ Check 29 OK:  全ての #snapshot-figure に本文段落が隣接")
if viol_ref:
    print("⚠️  Check 29b WARN: 隣接段落が図を名指ししていない #snapshot-figure が %d 枚（隣に本文はあるが、その図を論じていない疑い）" % len(viol_ref))
    for lno, img in viol_ref[:8]:
        print("   L%d: %s" % (lno, img))
    print("   → 図の隣接段落は「俯瞰図（下図）を見ると〜」「図中の〜」のように指示語で図を名指しし、その図から読み取った事実を最低1つ書く（common_framework.md §4 要点⑥）")
else:
    print("✅ Check 29b OK:  全ての図の隣接段落に図への言及（下図/図中 等）あり")
if viol_drill:
    print("⚠️  Check 29c WARN: 分析が薄いドリルダウン図が %d 枚（近傍±15行に 特許番号なし または 本文300字未満）" % len(viol_drill))
    for lno, img, ch, hp in viol_drill[:6]:
        print("   L%d: %s（本文%d字・特許番号%s）" % (lno, img, ch, 'あり' if hp else 'なし'))
    print("   → ドリルダウンは1枚ごとに ①どのクラスタを拡大したか ②内部で何が見えるか ③代表特許（公開番号）を本文で書く")
    print("   → data/saturnv_drilldown_*.json（サブクラスタ構造・代表特許入り）を必ず読んでから書く（deep_dive_guide.md Step 1 / map_reading.md §1）")
else:
    print("✅ Check 29c OK:  ドリルダウン図の分析量・代表特許引用あり（または該当図なし）")
PYEOF

# ==================================================================
# Check 30〜36: 構造化分析技法（ACH／リンチピン／ミラーイメージング）と裏付けの検査
# 設計原則（analysis/structured_techniques.md §0）: 機械の仕事は「正しさの判定」ではなく
# 「注意の誘導」。以下の検査は意味の正誤を裁かない。監査すべき箇所（対立仮説・弁別材料・
# 要の前提・裏付けのない断定）を人間の前に並べる。FAIL にするのは監査可能性そのものが
# 欠けている場合（ACH/リンチピンが1個も無い）に限る。
# ==================================================================

# 世代判定: セッション同梱テンプレートに competing-hypotheses 定義が無い＝構造化分析技法
# 導入前の旧スキーマ世代。旧セッションの gate 再実行を壊さないため Check 30/31 をスキップする。
STA_GEN=1
if [ -f "reports/report_style.typ" ] && ! grep -q "competing-hypotheses" "reports/report_style.typ" 2>/dev/null; then
  STA_GEN=0
fi

# Check 30: ACH（競合仮説）の有無（FAIL可・structured_techniques.md §1）
#   結論・提言章の主要結論に #competing-hypotheses（または中心的な問いの #ach-matrix）が
#   伴っているか。数えるのは有無であって質ではない（かかしは検出できない＝人間が監査する）。
echo ""
echo "--- Check 30: 競合仮説（ACH）の有無 ---"
if [ "$STA_GEN" -eq 0 ]; then
  echo "ℹ️  Check 30: 旧スキーマ世代のセッション（report_style.typ に competing-hypotheses 未定義）のためスキップ"
else
  ach_inline=$(count_matches '#competing-hypotheses\(' "$REPORT")
  ach_matrix=$(count_matches '#ach-matrix\(' "$REPORT")
  ach_total=$((ach_inline + ach_matrix))
  ccl_count=$(count_matches '#conclusion-box\(' "$REPORT")
  required=1
  if [ "$ccl_count" -ge 3 ]; then required=2; fi
  if [ "$ach_total" -eq 0 ]; then
    echo "❌ Check 30 FAIL: #competing-hypotheses / #ach-matrix が 0 個（主要結論に対立仮説の検討が一つも無い＝監査可能性の欠如）"
    echo "   → 結論・提言章の主要結論に、実際に支持のあったもっともらしい対立仮説と、それと食い違う弁別材料を書く"
    echo "   → 書き方・本ドメインの典型対立仮説（見かけの成長/リーダー断定/ホワイトスペース）: analysis/structured_techniques.md §1"
    fail=1
  elif [ "$ach_total" -lt "$required" ]; then
    echo "⚠️  Check 30 WARN: ACH = ${ach_total}個（inline=${ach_inline}・マトリクス=${ach_matrix}）。主要結論(conclusion-box=${ccl_count}個)に対し ${required}個以上を推奨"
  else
    echo "✅ Check 30 OK:  ACH = ${ach_total}個（inline=${ach_inline}・マトリクス=${ach_matrix}・主要結論=${ccl_count}個）"
  fi
fi

# Check 31: リンチピン（要の前提）の有無（FAIL可・structured_techniques.md §3）
echo ""
echo "--- Check 31: リンチピン（要の前提と崩れる条件）の有無 ---"
if [ "$STA_GEN" -eq 0 ]; then
  echo "ℹ️  Check 31: 旧スキーマ世代のセッションのためスキップ"
else
  lp_count=$(count_matches '#linchpin\(' "$REPORT")
  if [ "$lp_count" -eq 0 ]; then
    echo "❌ Check 31 FAIL: #linchpin が 0 個（主要結論の要の前提・崩れる条件が書かれていない＝監査可能性の欠如）"
    echo "   → 結論・提言章の冒頭の主要結論（1〜3個）に、要の前提・確認データ・この結論が崩れる条件を明示する"
    echo "   → 書き方: analysis/structured_techniques.md §3（overturns_if は観測できる具体的な出来事。空文禁止）"
    fail=1
  else
    echo "✅ Check 31 OK:  #linchpin = ${lp_count}個"
  fi
fi

# Check 30b/31b/33/35/36: 裏付け・非弁別・引用出所・断定の検査（すべて WARN・python で一括）
echo ""
echo "--- Check 30b/31b/33/35/36: 弁別材料・崩れる条件の裏付け／非弁別マトリクス／引用特許の出所／裏付けのない断定（WARN） ---"
STA_GEN="$STA_GEN" python3 - "$REPORT" <<'PYEOF'
import sys, os, re, json

raw = open(sys.argv[1], encoding='utf-8').read()
STA_GEN = os.environ.get('STA_GEN', '1')

# 出所をたどれる要素: 特許番号 / 数値＋単位 / 小数の指標値（HHI・Jaccard・中心性等） / #footnote
BACKED = re.compile(
    r'特開|特許第|特願|再表|WO\s?[0-9]{4}|US\s?[0-9]|EP\s?[0-9]|[0-9]{4}-[0-9]{5,7}'
    r'|[0-9][0-9,.]*\s*(件|社|％|%|倍|年|ポイント|pt|次元|語|回|位|セル|層)'
    r'|[0-9]+\.[0-9]+'
    r'|#footnote|HHI\s*[=＝]?\s*[0-9.]|CAGR\s*[+±\-−]?\s*[0-9.]')

def extract_args(name, key):
    """#name(...) ブロック内の key: "..." 文字列値を全て取り出す（括弧深度で終端判定）"""
    vals = []
    for m in re.finditer(r'#' + name + r'\(', raw):
        depth, i, start = 0, m.end() - 1, m.end() - 1
        while i < len(raw):
            if raw[i] == '(':
                depth += 1
            elif raw[i] == ')':
                depth -= 1
                if depth == 0:
                    break
            i += 1
        seg = raw[start:i]
        for vm in re.finditer(key + r'\s*:\s*"((?:[^"\\]|\\.)*)"', seg):
            vals.append(vm.group(1))
    return vals

# --- Check 30b: 弁別材料（counter_evidence）の裏付け ---
ces = extract_args('competing-hypotheses', 'counter_evidence')
bad_ce = [c for c in ces if not BACKED.search(c)]
if ces and bad_ce:
    print("⚠️  Check 30b WARN: 出所をたどれる要素（数値＋単位・特許番号・#footnote）を欠く弁別材料が %d/%d 件（手を振るだけの棄却＝かかしの疑い）" % (len(bad_ce), len(ces)))
    for c in bad_ce[:5]:
        print("   弁別材料: %s" % c[:60])
    print("   → 弁別材料は「その対立仮説と矛盾する具体的な材料」。HHI値・特許番号・件数等を明記する（structured_techniques.md §1-3）")
elif ces:
    print("✅ Check 30b OK:  全弁別材料（%d件）に出所をたどれる要素あり" % len(ces))
else:
    print("ℹ️  Check 30b: #competing-hypotheses なし（Check 30 参照）")

# --- Check 31b: 崩れる条件（overturns_if）の裏付け ---
ots = extract_args('linchpin', 'overturns_if')
bad_ot = [o for o in ots if not BACKED.search(o)]
if ots and bad_ot:
    print("⚠️  Check 31b WARN: 具体的な観測を欠く『崩れる条件』が %d/%d 件（『新しい情報が出たら』式の空文＝過信の兆候）" % (len(bad_ot), len(ots)))
    for o in bad_ot[:5]:
        print("   overturns_if: %s" % o[:60])
    print("   → 崩れる条件は観測できる具体的な出来事（例: 主要3極で権利化が進行していたら）を書く（structured_techniques.md §3-2）")
elif ots:
    print("✅ Check 31b OK:  全『崩れる条件』（%d件）に具体的な観測あり" % len(ots))
else:
    print("ℹ️  Check 31b: #linchpin なし（Check 31 参照）")

# --- Check 33: 非弁別マトリクス（×が皆無の #ach-matrix） ---
mats = []
for m in re.finditer(r'#ach-matrix\(', raw):
    depth, i, start = 0, m.end() - 1, m.end() - 1
    while i < len(raw):
        if raw[i] == '(':
            depth += 1
        elif raw[i] == ')':
            depth -= 1
            if depth == 0:
                break
        i += 1
    mats.append(raw[start:i])
if not mats:
    if STA_GEN == '0':
        print("ℹ️  Check 33: 旧スキーマ世代のセッションのためスキップ")
    else:
        print("⚠️  Check 33 WARN: #ach-matrix が無い（中心的な問い1つの仮説検証サマリを結論・提言章の冒頭または仮説検証サマリー章に置く。structured_techniques.md §2）")
else:
    nondiscr = [k + 1 for k, seg in enumerate(mats) if '"×"' not in seg and '×' not in seg]
    if nondiscr:
        print("⚠️  Check 33 WARN: ×（食い違い）が1つも無い #ach-matrix が %d/%d 枚（全て○の表は何も弁別していない＝体裁だけの水増し）" % (len(nondiscr), len(mats)))
        print("   → 判断を動かすのは対立仮説と食い違う材料だけ。×の無い表は材料の選び直しが必要（structured_techniques.md §2-2）")
    else:
        print("✅ Check 33 OK:  #ach-matrix %d枚すべてに×（弁別）あり" % len(mats))

# --- Check 35: 引用特許の出所（ミクロ分析節の番号が representative_patents.json 由来か） ---
RP = "reports/representative_patents.json"
if not os.path.isfile(RP):
    print("ℹ️  Check 35: %s なし（旧セッション or 未実行）。Phase C 開始時に capcom_schema/scripts/select_representatives.py を実行する" % RP)
else:
    rp = json.load(open(RP, encoding='utf-8'))
    allowed = set()
    for mod, dims in rp.items():
        if mod == "_meta":
            continue
        for recs in dims.values():
            for r in recs:
                for k in ("pub_number", "app_number"):
                    if r.get(k):
                        allowed.add(r[k])
    # ミクロ分析A（代表特許・象徴特許）の節だけを対象に番号を拾う。
    # ミクロ分析B（出願人プロファイル）や他章の根拠引用・ACH材料は条件検索による引用が
    # 許可されているため対象外（deep_dive_guide.md「決定的選定の正直な限界」）。
    micro_lines, in_micro = [], False
    for ln in raw.split('\n'):
        if re.match(r'^=+\s', ln):
            in_micro = (('ミクロ分析' in ln and 'ミクロ分析B' not in ln)
                        or '代表特許' in ln or '象徴特許' in ln)
            continue
        if in_micro:
            micro_lines.append(ln)
    cited = set(re.findall(r'[0-9]{4}-[0-9]{5,7}', '\n'.join(micro_lines)))
    rogue = sorted(cited - allowed)
    if not cited:
        print("ℹ️  Check 35: ミクロ分析節に特許番号の引用が見つからない（節見出しに『ミクロ分析』を含めているか確認）")
    elif rogue:
        print("⚠️  Check 35 WARN: ミクロ分析節に選定リスト外の特許番号が %d/%d 件（決定的選定の外から選んだ疑い＝なぜこの番号かを要確認）" % (len(rogue), len(cited)))
        for n in rogue[:8]:
            print("   リスト外: %s" % n)
        print("   → ミクロ分析A は reports/representative_patents.json の番号だけを引用する（deep_dive_guide.md ミクロ分析A）")
    else:
        print("✅ Check 35 OK:  ミクロ分析節の引用 %d件 すべて選定リスト由来" % len(cited))

# --- Check 36: 裏付けのない断定（事実の装いの検出） ---
# どの文が第1層（事実）かは機械判定できない（Check 14 がラベルを禁止しているため）。
# 危険度の高い断定の型を持つ文で、同文・隣接文に出所をたどれる要素が無いものを並べる。
# 誤検出（空振り）前提の注意の誘導であり、必ず WARN にとどめる。
# 行頭が # のもの（Typst 関数呼び出し: snapshot-figure / point-lead / chapter-summary 等）、
# 見出し、閉じ括弧行、コメントは地の文でないため対象外（誤検出の抑制）
SKIP = re.compile(r'^\s*(//|#[a-zA-Z]|=+\s|\]|\))')
ASSERT = re.compile(
    r'最大(?!限)|最多|最速|首位|唯一|支配的|最も(高|低|多|少|強|大き|小さ)'
    r'|が存在する|は見られない|は存在しない|皆無'
    r'|が原因で|を牽引して|に起因する|によって(増加|減少|拡大|縮小)した')
body_lines = [ln for ln in raw.split('\n') if ln.strip() and not SKIP.match(ln)]
sents = re.split(r'(?<=。)', ' '.join(body_lines))
sents = [s.strip() for s in sents if len(s.strip()) >= 15]
unbacked = []
for i, s in enumerate(sents):
    if not ASSERT.search(s):
        continue
    neighborhood = (sents[i - 1] if i > 0 else '') + s + (sents[i + 1] if i + 1 < len(sents) else '')
    if not BACKED.search(neighborhood):
        unbacked.append(s)
if unbacked:
    print("⚠️  Check 36 WARN: 裏付け（数値・特許番号・#footnote）が同文・隣接文に無い断定が %d 文（『事実の装い』の候補＝最も誤りが危険な箇所）" % len(unbacked))
    for s in unbacked[:8]:
        print("   断定: %s" % s[:70])
    print("   → 裏付けを添えるか、断定を解釈・推論の言い方（〜とみられる・〜と解釈できる）に落とす（common_framework.md §1 第1層の裏付けルール）")
else:
    print("✅ Check 36 OK:  裏付けのない断定なし")
PYEOF

# Check 32: ミラーイメージング点検（関係性立場のみ・WARN・structured_techniques.md §4）
echo ""
echo "--- Check 32: ミラーイメージング点検（対象企業から見た合理性の食い違い・WARN） ---"
if [ ! -f "$DECISIONS_FILE" ]; then
  echo "ℹ️  Check 32: $DECISIONS_FILE が無いためスキップ"
else
  mi_stance=$(python3 -c "import json; print(json.load(open('$DECISIONS_FILE')).get('narrative_stance',{}).get('code','') or '')" 2>/dev/null)
  case "$mi_stance" in
    competitor|buyer|supplier)
      mi_view=$(count_matches "から見ると|から見れば|の立場では|の視点では|にとっては合理" "$REPORT")
      mi_aspect=$(count_matches "補助金|サプライチェーン|供給網|既存顧客|規制|認証|地域戦略|既存事業" "$REPORT")
      if [ "$mi_view" -eq 0 ] || [ "$mi_aspect" -eq 0 ]; then
        echo "⚠️  Check 32 WARN: 立場=${mi_stance}(関係性立場) だが「対象企業から見た合理性の食い違い」の記述が見当たらない（視点対比表現=${mi_view}行・観点語=${mi_aspect}行）"
        echo "   → 自社から見ると不合理でも対象から見ると合理的な戦略を最低1点書く（観点: 補助金/サプライチェーン/既存顧客/規制・認証/地域戦略/既存事業。structured_techniques.md §4）"
      else
        echo "✅ Check 32 OK:  立場=${mi_stance} で視点対比=${mi_view}行・観点語=${mi_aspect}行"
      fi
      ;;
    *)
      echo "ℹ️  Check 32: 立場=${mi_stance:-未設定}（self/neutral）のため対象外"
      ;;
  esac
fi

# Check 34: 提言と ACH 結論の接続（WARN・structured_techniques.md §2-3）
#   提言・戦略章が「残った仮説」（＝ACH の結論）を前提として参照しているか。
#   切れて宙に浮いた提言＝ACH が飾りだった証拠を表に出す。妥当性は判定しない。
echo ""
echo "--- Check 34: 提言と ACH 結論の接続（WARN） ---"
if [ "$STA_GEN" -eq 0 ]; then
  echo "ℹ️  Check 34: 旧スキーマ世代のセッションのためスキップ"
else
  reco_refs=$(awk '
    /^=[[:space:]].*戦略的提言|^=[[:space:]].*戦略的インプリケーション/ { start=1; next }
    start && /^=[[:space:]]/ { exit }
    start { print }
  ' "$REPORT" | grep -cE "残った仮説|仮説検証サマリ|退けた対立仮説|最も矛盾の少ない解釈|採用した結論|検討した別解釈|比較検証" 2>/dev/null | head -n1)
  reco_refs=${reco_refs:-0}
  ach_any=$(count_matches '#competing-hypotheses\(|#ach-matrix\(' "$REPORT")
  if [ "$ach_any" -eq 0 ]; then
    echo "ℹ️  Check 34: ACH なし（Check 30 参照）のためスキップ"
  elif [ "$reco_refs" -eq 0 ]; then
    echo "⚠️  Check 34 WARN: 戦略的提言章が検証の結論（最も矛盾の少ない解釈）を参照していない（『仮説検証サマリ』『採用した結論』等への言及 0 行）"
    echo "   → 各提言は前提とする採用結論を名指しで置く（例:「実体は一社の急増（仮説検証サマリーで採用した結論）であるため、面での参入でなく個別対応を推奨」）"
    echo "   → 提言の前提ボックスの premise に採用結論を据え、overturns_if に「検討した別解釈が後から裏づけられたら」を書く（structured_techniques.md §2-3 三つの環）"
  else
    echo "✅ Check 34 OK:  戦略的提言章に検証結論への参照が ${reco_refs}行"
  fi
fi

# Check 37: 壁テキスト検出（構造化要素なしで地の文が長く連続する節・WARN）
#   「第1の発見は…第2の…」式の段落羅列が数千字続くと読めない。統合・サマリ系の節は
#   表（styled-table）・ボックス・小見出し・箇条書きで分節する（report_structure.md / common_framework.md §3）。
echo ""
echo "--- Check 37: 壁テキスト（構造化要素なしの地の文が3000字超連続・WARN） ---"
python3 - "$REPORT" <<'PYEOF'
import sys, re
lines = open(sys.argv[1], encoding='utf-8').read().split('\n')
# 構造化要素 = 分節として機能する行（表・図・ボックス・リスト・見出し）
STRUCT = re.compile(r'^\s*(#(styled-table|snapshot-figure|figure|image|ach-matrix|competing-hypotheses|'
                    r'linchpin|insight-box|note-box|point-lead|chapter-summary|conclusion-box|'
                    r'recommendation-card|kpi-dashboard|action-items|evidence-box|exec-summary|grid|table)'
                    r'|[-+*]\s|[0-9]+\.\s|=+\s)')
COMMENT = re.compile(r'^\s*//')
section = '(冒頭)'
acc = 0
walls = {}
for ln in lines:
    s = ln.strip()
    if re.match(r'^=+\s', s):
        section = re.sub(r'^=+\s+', '', s)
        acc = 0
        continue
    if s == '' or COMMENT.match(s):
        continue
    if STRUCT.match(s) or s in (']', ')', '),'):
        acc = 0
        continue
    acc += len(re.sub(r'\s', '', s))
    if acc > 3000:
        walls[section] = max(walls.get(section, 0), acc)
if walls:
    print("⚠️  Check 37 WARN: 構造化要素（表・図・ボックス・箇条書き・小見出し）なしで地の文が3000字超連続する節が %d 件（壁テキスト＝読了率が下がる）" % len(walls))
    for sec, ch in sorted(walls.items(), key=lambda x: -x[1])[:5]:
        print("   節「%s」: 連続 %d 字" % (sec[:40], ch))
    print("   → 統合・サマリ系の節は冒頭に統合表（パターン×発見×含意の styled-table）を置き、地の文の連続は4段落までを目安に、表・ボックス・小見出しで分節する（report_structure.md クロスモジュール章ルール）")
else:
    print("✅ Check 37 OK:  壁テキストなし")
PYEOF

# Check 16: PPTX（presentation.pptx）の機械チェック（任意出力。存在時のみ）
# slides_spec.md §0.9 のルールを AI 判断に頼らず機械的に検査する。
# 抽出は python-pptx で決定的に行う（再現性担保）。python-pptx 未導入時は WARN でスキップ。
echo ""
echo "--- Check 16: PPTX 機械チェック (reports/*.pptx・slides_spec §0.9) ---"
PPTX=$(ls reports/*.pptx 2>/dev/null | head -n 1 || true)
if [ -z "$PPTX" ]; then
  echo "ℹ️  Check 16: PPTX 未生成のためスキップ（PPTX は任意出力）"
else
  echo "ℹ️  Check 16 対象: $PPTX"
  python3 - "$PPTX" <<'PYEOF'
import sys, re
try:
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE
except Exception as e:
    print("⚠️  Check 16 WARN: python-pptx 未導入のため PPTX チェックをスキップ (%s)" % e)
    sys.exit(0)

path = sys.argv[1]
try:
    prs = Presentation(path)
except Exception as e:
    print("❌ Check 16 FAIL: PPTX を開けません（ファイル破損の疑い）: %s" % e)
    sys.exit(1)

# 出所欄に出してはいけない分析モジュール/ブランド機能名
MOD = re.compile(r'NEBULA|Saturn\s*V|MEGA|Explorer|CREW|EAGLE|ATLAS|\bCORE\b|TELESCOPE|PULSE|PROBE')
# 内部識別子（JSONファイル名・内部フィールド名・プロセス用語）
INTERNAL = re.compile(
    r'(saturnv_clusters|mega_momentum[a-z_]*|nebula_[a-z]+|atlas_statistics|explorer_[a-z]+|'
    r'eagle_[a-z]+|core_classification|crew_network)\.json'
    r'|\b(spatial_context|cluster_dynamics|noise_analysis|quadrant_summary|hype_cycle_phase)\b'
    r'|\bdeep[-_ ]dive\b|Phase\s+[ABCD]\b'
    r'|report(_executive|_style)?\.typ|reports/|capcom_schema/')  # 内部ファイルパスの出所漏れ（例「（出所）reports/report.typ」）
PLACEHOLDER = re.compile(r'xxxx|lorem|ipsum|プレースホルダ|ここに記載|TODO|FIXME', re.I)
# 事業・市場ファクト（特許データからは導けない外部事実）のマーカー
# 注: CAGR/件数/クラスタ等は特許データから算出する内部指標なので含めない（誤検出回避）
BIZ = re.compile(r'世界初|初採用|商用プラント|商用化|プレスリリース|市場規模|億ドル|兆円|社が採用|社に採用')
# 「企業名/固有名 + 数値 + 件」の抽出（数値一貫性チェック用）
NUMPAIR = re.compile(r'([一-龥ァ-ヴーA-Za-z]{2,8})[はがのもをにで、]{0,3}\s*([0-9０-９,，]+)\s*件')
# 16j で固有名扱いしない一般語（誤検出抑制。「累積460件/991件」のような集計語）
NUM_STOP = {'累積', '合計', '総計', '全体', '上位', '直近', '平均', '最大', '最小', '中央',
            '総数', '件数', '対象', '全件', 'うち', '前年', '前期', '今期', '当期', '計'}
# 過剰修辞・気取った比喩（terminology.md §6.5）
RHETORIC = re.compile(r'越境者|狼煙|地殻変動|試金石|旗手|号砲|幕開け|夜明け|胎動|風雲児|金字塔|羅針盤|苗床|三段ロケット|主役交代')

def _to_int(s):
    s = s.translate(str.maketrans('０１２３４５６７８９', '0123456789'))
    s = re.sub(r'[,，]', '', s)
    return int(s) if s.isdigit() else None

def shape_texts(slide):
    out = []
    for sh in slide.shapes:
        try:
            if sh.has_text_frame and sh.text_frame.text.strip():
                out.append(sh.text_frame.text.strip())
            if sh.has_table:
                for row in sh.table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            out.append(cell.text.strip())
        except Exception:
            pass
    return out

def has_picture(slide):
    for sh in slide.shapes:
        try:
            if sh.shape_type == MSO_SHAPE_TYPE.PICTURE:
                return True
        except Exception:
            pass
    return False

slides = list(prs.slides)
n = len(slides)
src_viol = []; internal_viol = []; placeholder_viol = []; footer_viol = []; thin = []
biz_src_viol = []          # 16i: 事業ファクトを特許データセット出所にしている面
num_map = {}               # 16j: 固有名 -> {数値: [slide,...]}
tilde_titles = []          # 16k: タイトルに波ダッシュ「～」副題（数値レンジは除外）
rhetoric_viol = []         # 16l: 過剰修辞・気取った比喩
types = []   # スライドタイプ推定（図形構成から機械的に分類）
for i, s in enumerate(slides):
    texts = shape_texts(s)
    joined = "\n".join(texts)
    pic = False; has_tab = False; n_auto = 0
    for sh in s.shapes:
        try:
            if sh.has_table:
                has_tab = True
            st = sh.shape_type
            if st == MSO_SHAPE_TYPE.PICTURE:
                pic = True
            elif st == MSO_SHAPE_TYPE.AUTO_SHAPE:
                n_auto += 1
        except Exception:
            pass
    # 16k: タイトル（先頭テキスト）に「空白+～」＝副題用法（数値レンジ 21〜24% は空白なしなので除外）
    if texts and re.search(r'[\s　]～', texts[0]):
        tilde_titles.append((i + 1, texts[0].replace("\n", " ")[:48]))
    mr = RHETORIC.search(joined)        # 16l: 過剰修辞
    if mr:
        rhetoric_viol.append((i + 1, mr.group(0)))
    src_labels = [t for t in texts if t.lstrip().startswith("（出所") or t.lstrip().startswith("(出所")]
    for t in texts:
        ts = t.lstrip()
        if (ts.startswith("（出所") or ts.startswith("(出所")) and MOD.search(t):
            src_viol.append((i + 1, t.replace("\n", " ")[:70]))
    # 16i: 事業ファクトを含むのに出所が特許データセットのみ（Web出所なし）
    patent_sourced = any("特許データセット" in s2 for s2 in src_labels)
    web_sourced = any(("付録C" in s2) or ("http" in s2) or ("Web" in s2) or ("取得日" in s2)
                      or (".jp" in s2) or (".com" in s2) for s2 in src_labels)
    if BIZ.search(joined) and patent_sourced and not web_sourced:
        m = BIZ.search(joined)
        biz_src_viol.append((i + 1, m.group(0)))
    # 16j: 「固有名+N件」を全スライドから収集（プロセス文＋表セル）
    for nm, num in NUMPAIR.findall(joined):
        v = _to_int(num)
        if v is not None and v > 0 and nm not in NUM_STOP:
            num_map.setdefault(nm, {}).setdefault(v, []).append(i + 1)
    for sh in s.shapes:                       # 表は「名前列＋最初の整数セル」をペア化
        try:
            if not sh.has_table:
                continue
            for row in sh.table.rows:
                cells = [c.text.strip() for c in row.cells]
                if len(cells) < 2:
                    continue
                nm = cells[0]
                if nm in NUM_STOP:
                    continue
                if not re.match(r'^[一-龥ァ-ヴーA-Za-z][一-龥ぁ-んァ-ヴーA-Za-z0-9（）()・\-]{1,15}$', nm):
                    continue
                for c in cells[1:]:
                    v = _to_int(c)
                    if v is not None and v > 0:
                        num_map.setdefault(nm, {}).setdefault(v, []).append(i + 1)
                        break
        except Exception:
            pass
    if INTERNAL.search(joined):
        internal_viol.append(i + 1)
    if PLACEHOLDER.search(joined):
        placeholder_viol.append(i + 1)
    if "APOLLO CAPCOM" in joined:
        footer_viol.append(i + 1)
    chars = len(re.sub(r'\s', '', joined))
    # 章扉(section_slide)識別: "01"〜"99" の2桁番号を単独テキストで持ち、図表なし・AUTO_SHAPE少・短い面。
    # add_section_slide は番号 f"{n:02d}" + 章名 + リード文で 60字前後あり、chars<40 では拾えないため番号で判定する。
    is_section = (not pic and not has_tab and n_auto <= 4 and chars < 130 and
                  any(re.match(r'^[0-9]{2}$', t.strip()) for t in texts))
    # タイプ推定: 表紙 / 図+注釈(picture) / 表 / 章区切り(薄・少図形) / ポンチ絵(多図形) / テキスト
    if i == 0:
        typ = "表紙"
    elif pic:
        typ = "図+注釈"
    elif has_tab:
        typ = "表"
    elif is_section or (chars < 40 and n_auto <= 5):
        typ = "章区切り"
    elif n_auto >= 7:
        typ = "ポンチ絵"
    else:
        typ = "テキスト"
    types.append(typ)
    # 薄さ判定: 表紙/締め/章扉は除外。コンテンツ面のみ、図/表あり=80字・図なし=150字を下限とする。
    # 図あり面でもリード文＋読み取り注釈が要る（旧基準「図なし40字未満」だと図あり薄面を見逃した）。
    # 良質デッキはコンテンツ面 中央333字・平均296字（リード文＋根拠箇条書き4-6点・§0.9-A0/B）。
    if i != 0 and i != n - 1 and typ not in ("表紙", "章区切り"):
        thin_threshold = 80 if (pic or has_tab) else 150
        if chars < thin_threshold:
            thin.append((i + 1, chars, thin_threshold))

fail = 0
print("ℹ️  Check 16: スライド %d 枚" % n)

# 16a: 出所に分析モジュール名（FAIL）
if src_viol:
    fail = 1
    print("❌ Check 16a FAIL: 出所欄に分析モジュール名が %d 件（出所はデータ名/Web実出所にする・§0.9-D）" % len(src_viol))
    for idx, txt in src_viol[:8]:
        print("     slide %d: %s" % (idx, txt))
else:
    print("✅ Check 16a OK:  出所欄に分析モジュール名なし")

# 16b: 内部識別子の露出（FAIL）
if internal_viol:
    fail = 1
    print("❌ Check 16b FAIL: 内部識別子(JSON名/フィールド名/プロセス用語)の露出 slide %s（terminology.md違反）"
          % ",".join(map(str, internal_viol[:12])))
else:
    print("✅ Check 16b OK:  内部識別子の露出なし")

# 16c: プレースホルダ残り（FAIL）
if placeholder_viol:
    fail = 1
    print("❌ Check 16c FAIL: プレースホルダ/TODO残り slide %s" % ",".join(map(str, placeholder_viol[:12])))
else:
    print("✅ Check 16c OK:  プレースホルダ残りなし")

# 16d: フッター「APOLLO CAPCOM」（FAIL）
if footer_viol:
    fail = 1
    print("❌ Check 16d FAIL: フッター/本文に 'APOLLO CAPCOM' が混入 slide %s（フッターは 'APOLLO' のみ）"
          % ",".join(map(str, footer_viol[:12])))
else:
    print("✅ Check 16d OK:  'APOLLO CAPCOM' の混入なし")

# 16e: 情報量の薄いコンテンツ面（図あり<80字 / 図なし<150字・表紙/締め/章扉除く）。薄い面が3割超で FAIL
if thin:
    content_n = sum(1 for ii, t in enumerate(types) if ii != 0 and ii != n - 1 and t not in ("表紙", "章区切り"))
    thin_ratio = (len(thin) * 100) // max(1, content_n)
    msg = ("情報量の薄いコンテンツ面が %d/%d 枚（図あり面<80字 or 図なし面<150字）。"
           "リード文＋根拠箇条書き4-6点で埋める（§0.9-A0/B・良質デッキは中央333字）" % (len(thin), content_n))
    if thin_ratio >= 30:
        fail = 1
        print("❌ Check 16e FAIL: " + msg)
        print("     → 薄い面がコンテンツ面の %d%%（3割超）＝デッキ全体が薄い。reports/report.typ の各章の論証を土台に各面を厚くする（初回生成で頻発する失敗）" % thin_ratio)
    else:
        print("⚠️  Check 16e WARN: " + msg)
    for idx, ch, th in thin[:10]:
        print("     slide %d: %d字 (要 %d字以上)" % (idx, ch, th))
else:
    print("✅ Check 16e OK:  情報量の薄いコンテンツ面なし")

# 16f: 枚数（WARN）
if n < 12:
    print("⚠️  Check 16f WARN: スライド %d 枚（最低12枚目安・推奨25-35枚。中身が薄くないか確認）" % n)
else:
    print("✅ Check 16f OK:  スライド %d 枚（12枚以上）" % n)

# 16g: 図(チャート)ありスライド比率 ≥ 50%（WARN）
pic_n = types.count("図+注釈")
visual_n = pic_n + types.count("表") + types.count("ポンチ絵")
textonly_n = types.count("テキスト")
pic_ratio = (pic_n * 100) // n if n else 0
visual_ratio = (visual_n * 100) // n if n else 0
if pic_ratio < 50:
    print("⚠️  Check 16g WARN: 図(チャート)ありスライド %d/%d=%d%% (<50%%目安)。"
          "視覚要素あり(図/表/ポンチ絵)=%d%%・テキストのみ=%d枚。"
          "スナップショット掲載かポンチ絵化を増やす" % (pic_n, n, pic_ratio, visual_ratio, textonly_n))
else:
    print("✅ Check 16g OK:  図ありスライド %d/%d=%d%% (≥50%%)・視覚要素あり=%d%%" % (pic_n, n, pic_ratio, visual_ratio))

# 16h: 同一スライドタイプ3枚連続の禁止（WARN）
runs = []
j = 0
while j < n:
    k = j
    while k + 1 < n and types[k + 1] == types[j]:
        k += 1
    if k - j + 1 >= 3:
        runs.append((types[j], j + 1, k + 1))
    j = k + 1
if runs:
    print("⚠️  Check 16h WARN: 同一タイプ3枚以上連続が %d 箇所（タイプを混ぜ単調さを避ける・Section4 多様性ルール）" % len(runs))
    for typ, a, b in runs[:8]:
        print("     slide %d-%d: 「%s」が %d 枚連続" % (a, b, typ, b - a + 1))
else:
    print("✅ Check 16h OK:  同一タイプ3枚連続なし")

# 16i: 事業・市場ファクトを特許データセット出所にしている面（WARN）
if biz_src_viol:
    print("⚠️  Check 16i WARN: 事業/市場ファクトを含むのに出所が特許データセットのみの面が %d 枚"
          "（世界初採用・商用化・市場規模等はWeb実出所＝付録C等を併記・§0.9-D）" % len(biz_src_viol))
    for idx, kw in biz_src_viol[:8]:
        print("     slide %d: 「%s」を含むが出所が特許データセット" % (idx, kw))
else:
    print("✅ Check 16i OK:  事業/市場ファクトの出所ミスなし")

# 16j: 同一固有名が複数の異なる「N件」で出現（数値不一致・WARN）
conflicts = {nm: d for nm, d in num_map.items() if len(d) >= 2}
if conflicts:
    print("⚠️  Check 16j WARN: 同一固有名が異なる件数で出現（数値の不一致 %d 件・全スライドで統一・§0.9-E）" % len(conflicts))
    for nm, d in list(conflicts.items())[:8]:
        parts = ["%d件(slide %s)" % (v, ",".join(map(str, sorted(set(sl))))) for v, sl in sorted(d.items())]
        print("     「%s」: %s" % (nm, " / ".join(parts)))
else:
    print("✅ Check 16j OK:  固有名+件数の不一致なし")

# 16k: タイトルに波ダッシュ「～」副題（ダサいので使わない・WARN）
if tilde_titles:
    print("⚠️  Check 16k WARN: タイトルに波ダッシュ「～」副題が %d 枚（1文で言い切る/補足は「—」か句点・§Section6）" % len(tilde_titles))
    for idx, t in tilde_titles[:8]:
        print("     slide %d: %s" % (idx, t))
else:
    print("✅ Check 16k OK:  タイトルの「～」副題なし")

# 16l: 過剰修辞・気取った比喩（WARN）
if rhetoric_viol:
    print("⚠️  Check 16l WARN: 過剰修辞・気取った比喩が %d 枚（平易・具体に言い換える・terminology.md §6.5）" % len(rhetoric_viol))
    for idx, w in rhetoric_viol[:8]:
        print("     slide %d: 「%s」" % (idx, w))
else:
    print("✅ Check 16l OK:  過剰修辞なし")

sys.exit(1 if fail else 0)
PYEOF
  pptx_rc=$?
  if [ "$pptx_rc" -ne 0 ]; then
    fail=1
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
