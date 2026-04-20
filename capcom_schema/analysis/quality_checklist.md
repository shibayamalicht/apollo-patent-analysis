# Phase D 品質検証ガイド

> このファイル名はレポート本文に書かないこと（執筆者の内部参照専用）。
> Phase D の概要 → `SKILL.md` Phase D セクション
> 用語統一ルール → `analysis/terminology.md`

---

## 1. 定量チェック（必ずコマンドで確認）

> **推奨**: 個別コマンドを1つずつ実行する代わりに、`bash capcom_schema/scripts/phase_d_gate.sh` を実行すれば全コマンドを統合実行し合否判定する(SKILL.md `## 0. 絶対遵守ゲートルール` に基づく強制ゲート)。

```bash
# 1. 行数チェック
wc -l reports/report.typ
# 合格基準: 最低800行。これ以下は内容不足

# 2. 代表特許引用数チェック
grep -c "特開\|特許第\|WO20\|JP20" reports/report.typ
# 合格基準: 15件以上

# 3. 4層モデルキーワードチェック
echo "=== Layer markers ===" && grep -c "解釈\|示唆\|と解釈" reports/report.typ && grep -c "にもかかわらず\|と合わせて\|洞察\|統合的に" reports/report.typ && grep -c "検討すべき\|推奨\|を検討\|参入\|投資" reports/report.typ
# 各Layer 5件以上が目安

# 4. クロスモジュール分析の分量チェック（修正: grep -A 50 では最大 51 行しか拾えないため awk で正しく章全体を抽出する）
awk '
    /^= .*クロスモジュール/ { start=1; next }
    start && /^= [^=]/ { exit }
    start { print }
' reports/report.typ | wc -l
# 合格基準: 80行以上（各パターン15-20行 x 最低3パターン + 統合考察）

# 5. snapshot-figure数チェック
grep -c "snapshot-figure" reports/report.typ
# 合格基準: 8枚以上

# 6. 用語統一チェック（全コマンドでヒット0が合格）
# 内部JSONファイル名の混入
grep -nE "(saturnv_clusters|mega_momentum|nebula_[a-z]+|atlas_statistics|explorer_[a-z]+|eagle_[a-z]+|core_classification|crew_network)\.json" reports/report.typ
# 内部フィールド名の混入
grep -nE "\b(spatial_context|cluster_dynamics|noise_analysis|quadrant_summary|time_series|hype_cycle_phase|macro_events|npl_gap_analysis)\b" reports/report.typ
# 内部ガイドファイル名の混入
grep -nE "(SKILL|common_framework|data_notes|cross_module|deep_dive_guide|report_structure|quality_checklist|terminology|map_reading|patent_citation|executive_summary_guide)\.md" reports/report.typ
# 内部プロセス用語（deep_dive / Phase A-D 等）の混入
grep -nE "\b(deep[-_ ]dive|Phase\s+[ABCD]\b|Phase\s+5a)\b" reports/report.typ
# J-PlatPat 等の具体DB名がユーザー未指定にも関わらず補われていないか
grep -n "J-PlatPat\|JPlatPat" reports/report.typ
# いずれも 0 ヒットが合格基準。1件でも残っていれば品質不合格。
```

---

## 2. 品質チェックリスト（全チェック項目）

以下の**必須項目**を全て確認する。

### 2.0 Phase A 母集団設計の読解（v8 追加、絶対遵守）

Phase A で以下 3 つの STOP-GATE を実施済みで、各々ユーザーが `AskUserQuestion` で明示的に応答していること。**AI の自己判断で省略したものは品質不合格**（SKILL.md `## 0. 絶対遵守ゲートルール` 第 1-2 項）。

#### 2.0.1 Phase A STOP-GATE A: `query_logic` 構造化読解

- [ ] `query_logic` が指定されていた場合、`analysis/query_logic_reading.md` §1 の 4 ステップ（DB 識別 → 構文分解 → 意図推定 → ユーザー確認）を実施した
- [ ] DB 識別結果・構文分解結果・各条件の意図推定を `AskUserQuestion` でユーザーに提示し、選択（進める／修正／補足追加）を取得した
- [ ] `query_logic` が未指定の場合、その旨をユーザーに 1 行で報告した

#### 2.0.2 Phase A STOP-GATE B: 意図 ↔ 論理 整合性検査

- [ ] `query_intent` と `query_logic` が両方指定されていた場合、`analysis/query_logic_reading.md` §4 の 8 項目チェックリストで対比した
- [ ] 検出した乖離を 3 段階（🔴 Critical / 🟡 Warning / 🔵 Info）に分類した
- [ ] Critical / Warning について、具体的な**改善提案**（例: 「検索式末尾に `* NOT (A23*/IC)` を追加」）を作成した
- [ ] 乖離報告 + 改善提案を `AskUserQuestion` で提示し、ユーザーが「修正 / 範囲と限界に明記 / 無視 / 乖離なし」のいずれかを選択した
- [ ] [B] 選択時（範囲と限界に明記）の場合、Phase D で「本分析の範囲と限界」章に乖離内容を明示的に記載している

#### 2.0.3 Phase A STOP-GATE C: データ側からの母集団実態確認 + 母集団タイプ判定

- [ ] **全ケースで実施**（`query_intent` / `query_logic` 未指定でも必須）
- [ ] `analysis/query_logic_reading.md` §5-1 の Level 2 項目（総件数・対象期間・上位 10 出願人・主要 IPC/FI・出願年分布・HHI・国/地域分布）を算出した
- [ ] 自動偏り警告閾値（上位 1 社 30% 超 / 上位 1 IPC 40% 超 / 直近 2 年 50% 超集中 / HHI > 0.25 / 特定国 95% 超）で異常を検出した
- [ ] **`analysis/population_type_metrics.md` §4-2 の判定マトリクスで母集団タイプ（A/A'/B/C/D）を推定した**
- [ ] タイプ判定結果と実態サマリを統合して `AskUserQuestion` で提示し、ユーザーが「タイプで進める / 別タイプに変更 / 範囲と限界に明記 / 再抽出」のいずれかを選択した
- [ ] 確定結果（タイプ + 禁止表現リスト + 偏り警告 + ユーザー決定）を `reports/_phase_a_decisions.json` に保存した

#### 2.0.4 設計意図（query_intent）の一貫性 — v8 新設

- [ ] Phase A STOP-GATE（サブクエスチョン化）を実施し、`_phase_a_decisions.json` の `sub_questions` フィールドに 3-5 個の観点（`id` / `content` / `keywords`）を保存した
- [ ] レポート本文に **問い/答え形式**（Q1 / A1、「問い 1 への回答」等）が**混入していない**ことを確認
- [ ] 以下 5 つの指定章で、**意図参照語**（「本分析の視座」「設計意図」「当初の分析観点」「本分析の視座に照らすと」等）が **各 1 箇所以上** 明示されている:
  - エグゼクティブサマリー冒頭
  - 各 `*_deep_dive.typ` の冒頭段落
  - クロスモジュール統合分析の各パターンの「結論」段
  - 戦略的提言（特に「分析結果の総括」冒頭）
  - 仮説検証サマリー
- [ ] `_phase_a_decisions.json` の各 `sub_questions.keywords` が結論章（総括＋戦略的インプリケーション＋推奨アクション）に **すべて登場** している
- [ ] Phase D gate **Check 12** が 0 件 FAIL で合格している

#### 2.0.5 分析過程で確認された追加的事項 章（条件付き必須）— v8 新設

- [ ] 以下のいずれかに該当する場合、レポートに「分析過程で確認された追加的事項」章が存在する（仮説検証サマリー後、戦略的提言前に配置）:
  - Phase A STOP-GATE B で Critical/Warning/Info 乖離があり、ユーザーが「このまま進めて明記」選択
  - Phase A STOP-GATE C でデータ偏り警告があり、同様の選択
  - Phase B/C/D の分析過程で設計意図外の重要観察があった
- [ ] 章タイトルが **「分析過程で確認された追加的事項」** と完全一致している（修辞的タイトル禁止）
- [ ] 各事項に「観察内容・設計意図との関係・本分析での扱い」の 3 点が記述されている
- [ ] 該当事項がない場合、`_phase_a_decisions.json` の `user_notes` に「追加的事項なし」と明記されている

#### 2.0.6 NEBULA 戦略判定（Phase A STOP-GATE D）— v8 新設

- [ ] `data/nebula_*.json` の存在を確認し、`data_available` を判定した
- [ ] データなしの場合、`AskUserQuestion` でユーザーに「Web 補完 / 省略」を選択させた（AI 自己判断禁止）
- [ ] 確定した `nebula_strategy`（`data_available`・`selected_mode`・`web_coverage_categories`・`notes`）を `reports/_phase_a_decisions.json` に保存した
- [ ] `web_compensation` モード時、Phase B の Web 調査で 4 カテゴリすべてをカバーするテーマを起草した
- [ ] `omit` モード時、「本分析の範囲と限界」章で特許情報のみ対象である旨を明記した
- [ ] Phase D gate **Check 13** がモード別検証を PASS している

#### 2.0.7 母集団タイプ別の指標・表現運用（Phase C/D 執筆時）

- [ ] Phase C/D 執筆時、`reports/_phase_a_decisions.json` の `population_type.code` を確認した
- [ ] タイプに応じて `analysis/population_type_metrics.md` §2 の指標解釈表を適用した:
  - タイプ **C**（単一企業）: 出願人 HHI を算出していない（無意味）。代替として発明者集中度を使った
  - タイプ **B**（競合限定）: 出願人 HHI / シェアを論じる際、「調査対象 N 社内での非対称性」と明記した
  - タイプ **D**（特定製品・技術テーマ）: 主語を「本テーマでは」に限定した
- [ ] タイプ B/C/D の場合、`_phase_a_decisions.json` の `forbidden_expressions` リストにある表現が本文中に混入していないか目視確認した
  - 例（タイプ B/C）: 「市場は寡占」「業界シェア」「市場構造」「競争環境」「業界全体で」等
- [ ] Phase D gate `Check 11` が `forbidden_expressions` の自動検出を実施、0 ヒットで合格することを確認した

### 2.1 構造・完成度

- [ ] **全モジュールdeep_dive**: `reports/` に最低4つの `*_deep_dive.typ` が存在するか（Saturn V, Explorer, MEGA, ATLAS）。各ファイルの最低行数（Saturn V 250行, Explorer 200行, MEGA 120行, ATLAS 120行, CORE 80行）を満たしているか
- [ ] **deep_dive存在チェック**: `ls reports/*_deep_dive.typ && wc -l reports/*_deep_dive.typ` を実行する。deep_diveが4つ未満、または各ファイルの行数が不足している場合は**Phase Cに戻る**

### 2.2 ミクロ分析

- [ ] **ミクロ分析**: 全deep_diveにミクロ分析A（代表特許引用）とミクロ分析B（企業戦略プロファイル）が含まれているか。代表特許は公開番号・タイトル・出願人の3点を含む形式で引用されているか
- [ ] **代表特許引用**: 各deep_diveのミクロ分析Aで合計15件以上の代表特許が引用されているか

### 2.3 情報ロス・転記品質

- [ ] **情報ロスチェック**: report.typの各モジュール章の行数が、対応するdeep_dive.typの行数の90%以上であるか（`wc -l` で比較）

### 2.4 分析品質

- [ ] **数値根拠**: 全主張に具体的数値が含まれているか
- [ ] **クロスモジュール分析**: 最低3パターン実施されているか（P1-P13から選択）
- [ ] **事実と推論の分離**: 4層モデルが明確に区別されているか
- [ ] **データソース明示**: 具体的なモジュール名・分析手法名を含むマーカーが付与されているか

### 2.5 可視化・画像

- [ ] **スナップショット画像（全章必須）**: **全てのdeep_dive.typおよびreport.typの全章（= 見出し単位）に最低1つの `#snapshot-figure()` が含まれていること**。チャート・マップの視覚的参照がない章は品質不合格とする。`ls snapshots/` で利用可能な画像を全て確認し、各章の分析内容に最も関連する画像を選択する。1つの画像を複数章から参照してもよい

### 2.6 AIインサイト・環境分析

- [ ] **AIインサイト反映**: prompts/から得た深い分析がレポートに反映されているか
- [ ] **環境分析の位置づけ**（`nebula_strategy.selected_mode` に応じて分岐）:
  - `execute`: NEBULA 環境分析章がエグゼクティブサマリーの直後に配置され、後続の各モジュール章が NEBULA 環境分析で導出された仮説を参照しているか
  - `web_compensation`: 「外部環境分析（Web 調査）」章が設置され、4 カテゴリ（市場規模 / 政策・規制 / 学術動向 / 主要企業動向）をすべてカバーし、各主張に `#footnote[...]` で出所明記されているか
  - `omit`: 環境分析章がなく、「本分析の範囲と限界」章で特許情報のみ対象である旨の注記があるか

### 2.7 新規分析要件

- [ ] **ノイズ分析（Saturn V）の萌芽テーマ考察**: Saturn V deep_diveにノイズ特許から抽出した萌芽テーマの考察が含まれているか
- [ ] **クラスタ動態マップの4象限解釈**: クラスタ動態マップ（累積件数xCAGR）の4象限が解釈されているか
- [ ] **多様性指標（Entropy/Gini）の解釈**: ATLAS deep_diveにHHI・Entropy・Giniの3指標組み合わせ分析が含まれているか
- [ ] **学術-特許クロス分析**（`nebula_strategy.selected_mode == "execute"` のみ必須）: NEBULA 学術クラスタ vs Saturn V 特許クラスタのクロス分析が実施されているか。モード `web_compensation` の場合は Web 調査の学術動向カテゴリとの照合で代替、モード `omit` の場合は省略可

### 2.7b スコープ限定ルール (terminology.md §6)

- [ ] **母集団スコープの明示**: レポート全体で「本母集団では〜」「本分析の特許群では〜」等の限定修飾が最低 5 件以上含まれているか。特に以下の箇所で義務:
  - エグゼクティブサマリー冒頭の対象範囲注記
  - 「本分析の前提」章の「本分析の範囲と限界」サブセクション
  - 各モジュール章の冒頭または最初の evidence-box
  - 結論章の各推奨アクション
- [ ] **無限化表現の抑制**: 「業界では〜」「市場では〜」「全体として〜」等の無限化語が、外部裏付け (`#footnote[...]`) なしに多用されていないか。phase_d_gate.sh の Check 10 で自動検出する (無限化語 ≤ 限定語 × 0.3 が合格基準)
- [ ] **別冊（経営層向け）の特別チェック**: 別冊生成フラグが ON の場合、別冊エグゼクティブサマリーに `#note-box[本分析の対象範囲: ...]` が含まれているか、各「発見」「推奨アクション」でスコープ明示が徹底されているか（詳細: `analysis/executive_summary_guide.md` §5-A）

### 2.8 結論・付録

- [ ] **結論章の完成度**: 戦略的提言章に4サブセクション（総括/インプリケーション/推奨アクション/アクションアイテム）が全て含まれ、推奨アクションが5件以上あるか
- [ ] **付録A（分析条件一覧）**: report.typの末尾に分析条件テーブル（特許データベース・対象件数・収録年情報・分析ツール・CAPCOM モジュール・パラメータ等）が含まれているか。`context.json` の値、および `population_meta.database_name` / `coverage_years` が正しく反映されているか
- [ ] **付録B（用語解説）**: report.typ中で使用した専門用語（SBERT/UMAP/HDBSCAN/CAGR/HHI/Entropy/Gini/c-TF-IDF/コミュニティ検出/媒介中心性/クラスタ動態マップ等）と、分析対象に固有の技術用語が解説されているか
- [ ] **付録D（母集団検索式、v8）**: `voyager/context.json` の `population_meta.query_logic` に値が入っている場合、付録 D「母集団検索式」セクションが追加され、検索式が `#raw(...)` ブロックで全文掲載されているか。`query_logic` が空文字の場合は付録 D セクションが存在しないか
- [ ] **Web調査出所（⚠️厳守、v8 で引用形式を一本化）**: Web調査を実施した場合、レポート中の全てのWeb由来情報はページ下部の脚注 `#footnote[...]` に出所（サイト名+URL+取得日）を載せる。本文中に旧形式の `_出所: ..._` インライン italic を書くのは禁止。以下のコマンドで確認する:
  ```bash
  # Web情報を使っている箇所の検出
  grep -cn "市場.*予測\|CAGR.*市場\|プレスリリース\|ニュース\|によると\|報道" reports/report.typ
  # #footnote[ 引用数
  grep -cn '#footnote\[' reports/report.typ
  # → Web情報使用数 ≤ #footnote 引用数 であること

  # 以前廃止した旧形式が残っていないか（0 が合格）
  grep -cn '_出所:\|_Source:' reports/report.typ
  ```
- [ ] **付録C（Web調査出所一覧）**: Web調査を実施した場合、付録にWeb調査出所一覧テーブル（No./情報内容/出所/取得日）が含まれているか
- [ ] **仮説検証サマリー**: レポート中で導出した全仮説（モード `execute` では NEBULA H1-H5、モード `web_compensation` では Web 調査由来の仮説、モード `omit` では特許データ由来の仮説のみ）が仮説検証サマリー章で回収されているか。判定（✅支持/❌棄却/⚠️部分支持/❓未検証）と根拠が明記されているか
  ```bash
  # 仮説の導出箇所と検証箇所の対応チェック
  grep -n "仮説\|H[1-9]\|と推察\|可能性がある\|と考えられる" reports/report.typ | wc -l
  grep -n "✅\|❌\|⚠️\|❓\|支持\|棄却\|未検証" reports/report.typ | wc -l
  # → 仮説導出数 ≈ 検証数であること
  ```

---

## 3. 推奨項目

以下は必須ではないが、レポート品質を向上させる推奨チェック項目:

- [ ] **可視化読解**: スナップショット画像の分析が表面的でなく、5ステップ読解（→ `analysis/map_reading.md`）が適用されているか
- [ ] **矛盾の指摘**: モジュール間のデータに矛盾がある場合、その理由を考察しているか
- [ ] **Mission Objective回答**: レポートがMission Objectiveの質問に明確に回答しているか
- [ ] **提言の具体性**: 推奨アクションが抽象的でなく、具体的な行動（技術名・企業名・時間軸）を含んでいるか
- [ ] **Evidence網羅性**: Evidence総数の半数以上が分析に活用されているか

---

## 4. 不合格時の対応フロー

1. 不合格項目があれば、該当セクションを修正する
2. deep_diveが4つ未満 or 行数不足 → Phase C に戻る
3. 情報ロス（report.typ < deep_dive.typ の90%）→ 省略箇所を復元する
4. 代表特許15件未満 → patents.csv を再検索して補完する
5. snapshot-figure 8枚未満 → `ls snapshots/` で画像を確認し追加する
6. 修正後、定量チェックを再実行して合格を確認する

→ 関連ファイル: `analysis/common_framework.md` セクション4（品質チェックリスト詳細）
