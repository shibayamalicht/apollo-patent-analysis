# PPTスライド仕様書 v6.0 "Mission Deck" — コンサルティングレポート品質 + ポンチ絵完全対応

python-pptx + Pillow で産業調査レポート品質のスライドを生成する仕様書。
**チャート/図が主役、テキストは注釈。タイトル＝結論を言い切る文。リード文はタイトル下の短下線＋プレーン文。**
**2層サーフェス（オフホワイト地＋白角丸カード）。色は「意味」にだけ使い、文字には載せない。**
**Noto Sans JP フォント統一（多段ウェイト運用）。全runに lang="ja-JP" 明示。裏表紙に Thank You は置かない。**
**テキストのみスライド ≤ 10%。ポンチ絵パターンで全スライドに視覚要素を保証。**

> **本仕様書は「設計ガイド」**。ヘルパー本体（フォント・色・レイアウトの実装コード）は実モジュール
> **`capcom_schema/templates/apollo_slides.py`** に抽出済み（import 可・全 `add_*_slide` と
> コア関数を内蔵）。生成スクリプトは**このモジュールを import して使う**（→ Section 2「import 運用」）。
> 本仕様書はコードをコピーするためのものではなく、**いつ・どのヘルパーを・どんな主張骨格で使うか**を
> 規定する。承認済みデザインの参照実装は `design_preview/generate_design_sample.py`。

**必要パッケージ:** `pip install python-pptx Pillow`

### v6.0 の変更点と不変点（最初に把握する）

**変更（デザイン刷新）**: ①2層サーフェス（BG オフホワイト地＋白角丸カード）とデザイントークン新設（Section 1）／②フォント階層の更新（Bold 主体。Black は章扉ゴースト数字のみ）／③リード文の体裁変更（■＋背景箱 → タイトル下の短下線＋プレーン文）と見出し規定の一本化（原則4）／④新部品（結論バンド面 `add_bands_slide`・注意/決め手ボックス `add_callout_box`・注釈カードレール `add_annotation_cards`・KPIカード/テーブル判定色の新体裁）／⑤表紙・章扉・裏表紙の新構成（裏表紙 Thank You 廃止）とフッター新形式（`APOLLO｜タイトル｜日付`＋頁番号）。

**不変（v5 から一切変えない）**: 情報密度基準（本文 250〜400 字/面・注釈 4〜6 項目）／比率規則（25枚以上・チャート+注釈 50%以上・テキストのみ 10%以下・章扉 15%以下）／Phase D `Check 16` 連動の判定規律／コネクタ（`p:cxnSp`）禁止／ドーナツ blockArc の ×0.6 換算／タイトル→リード→コンテンツの戻り値チェーン／**GroupShape（グループ化図形）不使用**／波ダッシュ「～」禁止／出所ルール（§0.9-D）。

---

## Section 0: 運用ガイド（最初に読む）

> この仕様書は **APOLLOユーザーが Claude Code / Claude Desktop に「APOLLOのスライドを作って」と依頼したときの実装手順書** として単体で完結する。本仕様書だけを読めばスライド生成が実行できるよう全情報を内包している。

### 0.1 起動条件
以下のいずれかでこの仕様書に従ってスライドを生成すること:
- ユーザーが「APOLLOでスライドを作って」「PPTを生成して」「プレゼン資料を作って」と依頼
- ユーザーが `/pptx` コマンドを実行（Claude Code環境）
- ユーザーが APOLLO の CAPCOM セッションフォルダ（または ZIP 展開済みフォルダ）を提示してスライド化を依頼

### 0.2 前提条件
- CAPCOM セッションフォルダ（`output/session_YYYYMMDD_HHMMSS/` または ZIP を展開したフォルダ）が存在すること
- 同階層の `capcom_schema/templates/apollo_template.pptx` をテンプレートとして使用すること
- 実行環境に `pip install python-pptx Pillow` が済んでいること

### 0.3 入力データの確認手順（着手前に必ず実施）
セッションフォルダで以下を確認し、利用可能なデータに応じてスライド構成を決める。

**⚠️ 最重要の役割分担**: スライドは **完成レポート `reports/report.typ` を「何を言うか（主張・論理・物語）」の一次ソース** とし、その論証を凝縮して図と文で再構成する。`data/*.json` は **正確な数値**、`snapshots/*.png` は **図版**、`voyager/*` は **Mission と画像指示** を供給する補助ソースである。**evidence の短い `description` を寄せ集めて作ってはいけない**（→ §0.9）。

| 確認対象 | 用途 |
|---------|------|
| **`reports/report.typ`（最優先・必読）** | **デッキの論証・要点・物語アーク・結論の一次ソース。各章の主張→根拠→示唆の連鎖をスライドへ凝縮する（§0.9）。`reports/report_executive.typ` があれば要約版の骨子としても活用** |
| `voyager/mission.json` | Mission Objective を表紙・サマリーに反映 |
| `data/patents.csv` | 出願人上位・クラスタ別件数・年別件数の把握 |
| `data/atlas_statistics.json` | ATLAS スライド（出願トレンド・多様性指標 HHI/Entropy/Gini） |
| `data/saturnv_clusters.json` | Saturn V スライド（クラスタ・ノイズ分析・クラスタ動態マップ） |
| `data/mega_momentum_<軸>.json`（applicant/ipc/fterm） | MEGA スライド（PULSE 4象限・軸別。各軸を個別に） |
| `data/explorer_*.json` | Explorer スライド（共起ネットワーク・急上昇キーワード） |
| `data/nebula_hype_cycle.json` | NEBULA スライド（ハイプサイクル） |
| `data/nebula_academic_clusters.json` | NEBULA スライド（学術ランドスケープ・学術クラスタ動態） |
| `snapshots/*.png` | チャート + 注釈スライドに優先掲載 |
| `prompts/` 配下の AI インサイト | 注釈・リード文生成の根拠（最低3件読了推奨） |

データが欠落しているモジュールはスキップしてよい。最低限必須: タイトル + KPI + ATLAS(1枚) + Saturn V(2枚) + MEGA(1枚) + 仮説検証 + クロージング ≒ 12枚（Section 5参照）。

### 0.4 実装手順
1. **本仕様書（slides_spec.md）を Section 1〜6 まで通読する** — 設計原則・デザイントークン・コアユーティリティ・スライドタイプ・推奨シーケンスを把握
2. **🔑 完成レポート `reports/report.typ` を通読する（最優先・§0.9）** — デッキはレポートの論証の凝縮版。各章の「主張・根拠（数値）・示唆（So what）・章どうしの繋がり」を把握し、**スライドの物語アーク（章立て）をレポートに合わせる**。`report_executive.typ` があれば要約版の骨子に使う
3. セッションフォルダの `data/` `snapshots/` `voyager/mission.json` `voyager/context.json` を確認し、利用可能データを判定
   - **`context.json` の `report_directives.image_slide_instruction` を必ず読む**: 値があれば「どの画像をどのスライドで使うか」をユーザー指示として最優先で反映（例:「表紙にクラスタ動態マップ」「権利化率マップはスライド必須」）。空ならAIが最適選択
   - `snapshots/*.png` はクリーン版（要点・出典なし）。**スライドは各コンテンツ面にリード文（要点）が必須**。要点は **②のレポートの該当章** から作る（数値は `data/*.json` で裏取り）。evidence の `description` だけで埋めない
4. **Section 5「推奨スライドシーケンス」** をベースに 25〜38 枚で構成決定（データが少ない場合は 12〜20 枚）。**構成はレポートの章順（前提→エグゼクティブサマリー→環境→俯瞰→動態→競争→クロス統合→仮説検証→結論・提言→将来）に沿わせ、モジュール羅列にしない**
5. **🔑 ヘルパーモジュール `capcom_schema/templates/apollo_slides.py` を import して生成スクリプトに使う（最重要）**。`import sys; sys.path.insert(0, "capcom_schema/templates"); from apollo_slides import *` の1行で全ヘルパーが使える（→ Section 2「import 運用」）。`_apply_font` / `add_title_shape` / `add_sub_message` / `add_title_slide` / `add_section_slide` / `add_chart_text_slide` / `add_kpi_slide` / `add_cards_slide` / `add_matrix_2x2_slide` / `add_arrow_flow_slide` / `add_donut_slide` / `add_issue_tree_slide` / `add_process_slide` / `add_stepup_slide` / `add_compare_slide` / `add_table_slide` / `add_progress_bar_slide` / `add_triangle_slide` / `add_pyramid_slide` / `add_hypothesis_slide` / `add_timeline_slide` / `add_shift_slide` / `add_convergence_slide` / `add_bands_slide` / `add_priority_actions_slide` / `add_action_items_slide` / `add_closing_slide` / `add_annotation_cards` / `add_callout_box` / `add_bottom_bar_and_footer` ほか を呼び出してスライド生成。
   - ⚠️ **自前で pptx のフォント・色・レイアウトを書き起こさないこと**。フォント（**Noto Sans JP**）・多段ウェイト（タイトル/KPI数値/バンド見出し=Bold・カード見出し=SemiBold・リード/チップ=Medium・本文=Regular・出所/フッター=Light）・2層サーフェス（白角丸カード）・短下線・フッターは**すべて `apollo_slides.py` のヘルパーに内蔵**されている。ヘルパーを使わず独自実装すると、単一ウェイトの平板なスライドになり品質が大きく劣化する
6. **タイトル → リード文 → コンテンツは必ず戻り値を連鎖させる**（0.5節参照）
7. **品質チェック**（0.7節）を全項目クリアしてから出力
8. 出力先: `reports/apollo_report_YYYYMMDD.pptx`

### 0.5 レイアウト連動ルール（最重要・厳守）
タイトル・リード文・コンテンツの y 座標を**ハードコード禁止**。`add_title_shape()` と `add_sub_message()` の戻り値を連鎖させる。

```python
# ✅ 正しい使い方: 戻り値の連鎖でタイトル長に追従
sub_y = add_title_shape(slide, "上位5クラスタが全体の58%を占有。技術集中化が加速")
content_y = add_sub_message(slide, "クラスタ0「CNF強化ゴム」が最大（48件）...", y=sub_y)  # 先頭に■を付けない（v6はプレーン文で描画）
# content_y を起点にチャート・カード・テーブル等を配置

# ❌ 間違い: y 座標をハードコードするとタイトルが長いとき重なる
add_title_shape(slide, "長いタイトル...")
add_sub_message(slide, "...", y=0.90)
```

`add_title_shape()` はタイトル長に応じてフォントサイズと高さを動的調整し、タイトル直下に **ACCENT 短下線**を描いてリード文開始 y を返す。`add_sub_message()` は**プレーンなリード文**（背景箱・■なし）を描き、コンテンツ開始 y を返す。

### 0.6 トークン効率の注意（厳守）
- **サブエージェント（Agent tool）を起動しない** — 全処理をメインコンテキスト内で完結
- 本仕様書は **一度だけ読み**、以降は会話内で参照する（再読み込みしない）
- `snapshots/` の画像は **スライドに使うものだけ** 読み込む
- セッションデータ（`data/*.json`、`patents.csv`）は **必要範囲のみ抽出** する。全件 dump は禁止

### 0.7 品質チェックリスト（出力前に全項目確認）
- [ ] フッターが**新形式**（左 `APOLLO｜レポートタイトル｜日付`・右 頁番号・上にヘアライン。`add_bottom_bar_and_footer()` 使用。ブランドは "APOLLO" — "APOLLO CAPCOM" は不可）
- [ ] 全スライドのタイトルが**結論型**（原則4。ラベル型禁止、数値を1つ以上含む）。**波ダッシュ「～」は使わない**（補足は「—」か句点で短い2文に）
- [ ] **チャート + 注釈スライドが全体の50%以上**（Section 1 原則5）
- [ ] **テキストのみスライドが全体の10%以下**（エグゼクティブサマリー + 結論のみに限定）
- [ ] 全分析スライドに**視覚要素**（チャート/画像/ポンチ絵）が含まれる
- [ ] 全 run に **Noto Sans JP** + `lang="ja-JP"` 設定（`_apply_font()` 経由）。役割別ウェイト（タイトル・KPI数値・バンド見出し=Bold／カード見出し=SemiBold／リード・チップ=Medium／本文=Regular／出所・フッター=Light。**Black は章扉ゴースト数字のみ**）を使い分けている
- [ ] **各コンテンツ面が「主張骨格」を満たす**（§0.9-A0）: ①アイブロウ ②結論を言い切る主張見出し ③**リード文（核心主張の完結文・40〜90字）** ④根拠（図/カード/数値） ⑤**締め文（So What の地の文一文・30〜70字）**。リード文・締め文を欠く断片箇条書きだけの面は不可
- [ ] **各コンテンツ面の情報密度が十分**（§0.9-B）: 本文（リード文＋注釈）合計 **250〜400 字**・根拠注釈 **4〜6 項目**（良質デッキ実測 中央 333 字）。**図がある面も本文を薄くしない**（リード文＋読み取り注釈4-6点を添える）。薄い面がコンテンツ面の3割超で Phase D `Check 16e` FAIL
- [ ] **リード文＝核心主張の完結した一文**（40〜90字・2行以内・数値込み。体言止め・単語列でなく文で言い切る。v6 はプレーン文で描画され■・背景箱は付かない — `message` に■を含めない）
- [ ] 注釈は**断片の羅列でなく「数値を含む完結した主張文」**（5語の事実断片を並べない）。箇条にする場合も各項目1〜2行の完結文（§0.9-A0/A）
- [ ] タイトルとリード文が**重なっていない**（戻り値連鎖を使用）
- [ ] カラーは **Section 1 のデザイントークンのみ使用**（BG/CARD/INK/DEEP/CAT/TINT/CHIP/判定色ほか）。**色は意味にだけ使い、CAT 彩色を文字に載せない**（識別は常にラベル併記）
- [ ] 表紙に **Mission Objective** 記載（v6 表紙構成 → Section 3）
- [ ] エグゼクティブサマリーに **KPI 3〜4 個**（`add_kpi_slide()`）
- [ ] 戦略提言は**矢印プロセスフロー or ステップアップ**（`add_process_slide()` / `add_stepup_slide()`）でポンチ絵化、**結論の層構造は結論バンド面**（`add_bands_slide()`）
- [ ] 仮説検証スライド（`add_hypothesis_slide()`）に各仮説の判定と根拠（判定色: 支持=GOOD_TX/部分支持=PART_TX）
- [ ] **章区切りスライドに「この章の結論メッセージ」（20-40字）を必ず配置**（巨大番号＋章題だけの空白仕切りは禁止）
- [ ] **章区切りスライドはデッキの15%以下**（章扉を量産して薄さを生まない）
- [ ] **表スライドに「読み取り（結論）」1行＋示唆2-3点を必ず併記**（表の置きっぱなし禁止。決め手は `add_callout_box()` で明示）
- [ ] **表紙・締めにも余白を埋める要素**を配置（表紙=Mission＋リード段落＋KPIカード＋主図版、締め直前=アクションアイテム面。裏表紙は v6 定型で **Thank You を置かない**）
- [ ] **コネクタ（`p:cxnSp`）不使用**。流れ・矢印は CHEVRON / ARROW オートシェイプで描く（コネクタは破損要因）
- [ ] **GroupShape（グループ化図形）不使用**。部品は個別シェイプで直接配置する（グループ化は座標ズレ・破損要因）
- [ ] **レポート（`reports/report.typ`）を土台に作成した**（evidence の短い説明文の寄せ集めでない・§0.9）。各スライドが主張→根拠（数値）→示唆の最小ロジックを運んでいる
- [ ] **章立てがレポートの物語アークに沿う**（モジュール羅列でない）。クロス統合・仮説検証・結論/提言は複数モジュールを束ねた面になっている
- [ ] **出所が分析モジュール名でない**（`（出所）NEBULA …` は不可）。特許データ由来は「特許データセット」、Web由来は実出所（サイト/URL/取得日）。モジュール名は本文側で使用（§0.9-D）
- [ ] **事業・市場ファクト（世界初採用/商用化/市場CAGR・億ドル/プレスリリース）の出所が実出所（付録C等）**になっている（特許データセット出所のままにしない・§0.9-D）
- [ ] **同一数値・固有名がデッキ全体で一貫**（件数・%・年・順位の食い違いなし。例「旭化成91件/92件」混在は不可・§0.9-E）
- [ ] **章扉の結論に「結論：」ラベルを付けず地の文**。締めブランドは「APOLLO」で統一（§0.9-F）

### 0.8 改善ルール — 「薄さ」と「作図の貧弱さ」の是正（最優先・厳守）

過去の自動生成デッキで頻発した品質劣化（実測で確認）を是正するための必須ルール。**(a) 章扉・表紙・締めが空白だらけ（デッキの約4割が薄い面）、(b) 表に結論が無く読者任せ、(c) 作図が「色付き四角の羅列」止まりで矢印・流れ・ロジック図が皆無**、の3点を撲滅する。

#### A. 余白を作らない（Fill the slide）— 薄いスライドの撲滅
- **章区切りスライドに必ず「この章の結論メッセージ」（その章で最も言いたい結論、20-40字）をテキストで配置する**。巨大番号＋章題だけの"仕切りだけ"スライドは禁止。可能なら「この章で答える問い」も1行添える。
- **章扉はデッキの 15% 以下**。章が6つあっても章扉は主要章のみに絞り、量産しない。
- **表紙**は v6 構成（Mission Objective のサブタイトル＋リード段落＋KPIカード3枚＋主図版カード → Section 3）で版面を埋める。**締め**は裏表紙の直前に `add_action_items_slide()` で次アクションを列挙し、裏表紙自体は v6 定型（タイトル＋ブランド行＋日付。Thank You・飾り文言なし）で静かに閉じる。
- 各コンテンツ面は「タイトル＋リード文＋（図 or ポンチ絵）＋注釈4-6点」で**版面の8割以上**を使う。逆に詰め込み過ぎ（はみ出し）も禁止（戻り値連鎖で配置）。

#### B. 表スライドには必ず「読み取り（結論）」を添える
- `add_table_slide()` で表だけを置いて読者任せにしない。表の上に **リード文（結論）1行**、表の周辺に **「この表から何を読むか」を注釈で2-3点** 必ず付す。判定の分かれ目は **`add_callout_box()` の「決め手」ボックス**で言語化する。
- とくに「権利化率表」「仮説検証表」「将来見通し表」は、数字の羅列でなく**示唆を言語化**する（例:「大王製紙は出願量首位だが権利化率は中央値以下＝量先行・質限定」）。

#### C. ポンチ絵を「色付き四角の羅列」で終わらせない
- 2×2マトリクスは、ただの4分割矩形にせず、**縦横の軸（矢印＋軸ラベル「← 成長率 →」等）と各象限の意味ラベル**を必ず描く。→ **専用ヘルパー `add_matrix_2x2_slide()` を使う（Section 3 カタログ・軸の矢印・低/高・象限ラベルを自動描画）**。
- プロセス・因果・ロジックの流れは **矢羽根（CHEVRON）/ 矢印（ARROW）オートシェイプ**で方向を示す。→ 横向きの流れは **専用ヘルパー `add_arrow_flow_slide()`**、論点分解は **`add_issue_tree_slide()`（ロジックツリー）** を使う。
  - ⚠️ **コネクタ（`p:cxnSp` / connector）は使わない**（python-pptx でファイル破損を起こす既知の不具合）。矢印は必ず CHEVRON / ARROW 等の**オートシェイプ**で描画する。**GroupShape も使わない**（個別シェイプを直接配置）。
- 構成比（クラスタ別件数・出願人シェア等）は **ドーナツ図 `add_donut_slide()`（`BLOCK_ARC` で描画）** を使う。色付き矩形の羅列で代用しない。
- 章ごとに**矩形以外の図形・方向表現を最低1つ**使う（マトリクス `add_matrix_2x2_slide()`／矢羽根フロー `add_arrow_flow_slide()`／ドーナツ `add_donut_slide()`／ロジックツリー `add_issue_tree_slide()`／トライアングル `add_triangle_slide()`／ピラミッド `add_pyramid_slide()`／プロセス `add_process_slide()`／ステップアップ `add_stepup_slide()`／タイムライン `add_timeline_slide()` 等を積極活用）。
- チャートは **APOLLO アプリが書き出した高解像 PNG（snapshots/）を白カードに載せて貼る運用で良い**（ネイティブグラフ化は不要・APOLLO の設計通り）。ただし**貼った図には必ず読み取り注釈を併記**し、図の置きっぱなしにしない。図＋注釈面の右レールは **`add_annotation_cards()` の4部構成（図の見方/読み取り/別の見方/示唆）を推奨**（§Section 3）。

#### D. 参考（コンサル品質の python-pptx 実装テクニック）
本仕様の型を磨く際の参考。原則（**タイトル＝結論・余白禁止・矢羽根で流れ・表に結論・同一行はフォント揃え**）を取り込む:
- likaku/Mck-ppt-design-skill（70レイアウト、「余白禁止」「コネクタ禁止→矢羽根代替」「象限は厳密に4要素」）: https://github.com/likaku/Mck-ppt-design-skill
- sruthir28/enterprise-ai-skills（Issue Tree / Minto Pyramid / SCPR 等の論証骨格、棒＋コールアウト等）: https://github.com/sruthir28/enterprise-ai-skills
- seulee26/mckinsey-pptx（テーマ集中管理、章扉に意味を持たせる構造系テンプレ）: https://github.com/seulee26/mckinsey-pptx

### 0.9 レポートを土台にする — 「薄い・事実列挙」の根治（最優先・厳守）

過去デッキの最大の品質問題は **「完成レポートを土台にせず、各モジュールのスナップショットと短い説明文を寄せ集めて1枚ずつ並べていた」** こと。結果、(i) 本文が短文の事実断片の羅列になり、(ii) レポートが持つ論理（なぜそう言えるか）が運ばれず、(iii) 構成がモジュール羅列になっていた。これを根治するための必須ルール。

#### A0. スライドの主張骨格 — 箇条書きでなく「内容のある文」で論じる（最優先・全コンテンツ面で厳守）

優れたコンサル/エディトリアル・デッキは、5語の断片を箇条で並べず、**1スライド＝1つの主張を完結した文で論証**する。各コンテンツ面は必ず次の**5層の骨格**で構成する（v6 でも骨格は不変。変わったのはリード文の体裁のみ）:

1. **アイブロウ**（モジュール/章名・小さく）— 例「NEBULA / 環境分析」「ATLAS / 基本統計」。`add_title_shape(slide, title, eyebrow="NEBULA / 環境分析")` の `eyebrow` 引数で添える（ゴシック・10pt・字間広め・ミュート色。明朝/等幅は使わない）。
2. **主張見出し**（そのスライドで言いたい結論を**言い切る文**で・**原則4に従う。見出し規定は原則4が唯一の正**）— 例「俯瞰図: 知財は『本体・製造・制御』の三本柱に集約」「出願の厚みは2015年以降に形成 — 直近の減少は見かけ」。ラベル（"出願トレンド"）や短い名詞句（"競争構造の評価"）は不可。
3. **リード文**（核心の主張を1つの完全な文で・40〜90字）— タイトル直下の **ACCENT 短下線に続くプレーン文**として置き（`add_sub_message`。■・背景箱は使わない）、**数値を織り込んだ完結文**にする。体言止め・単語列は不可。
   - 例「出願の単調増加と高いノイズ率（34.2%）は、本母集団が黎明期を脱し、用途多様化を伴う成長加速期にあることを示す。」
4. **根拠**（KPIカード／バンド／注釈カード／象限／実マップ PNG）— ラベル＋数値＋短い説明でデータを"見せる"層。
5. **締め文**（So What を1文で・30〜70字）— その根拠から導かれる含意・行動示唆を**地の文の一文**で。
   - 例「単調増加は特定企業の戦略ではなく、技術領域全体の裾野が時間とともに広がってきた帰結。」

**箇条書きを使う場合も、各項目は「5語の断片」でなく「数値を含む完結した主張文」**にする（§A の良い例参照）。**リード文（核心主張）と締め文（So What）が無く、断片箇条書きだけで終わるスライドは不可**（Phase D `Check 16/19` で検出 FAIL）。

**悪い例（断片の羅列・避けるべき型）** ❌
- ・CAGR 12.4%／・クラスタ24／・ノイズ34.2%／・出願人689社  ← ラベルと数字だけで主張が無い
**良い例（主張見出し＋リード文＋根拠＋締め文）** ✅
- 主張見出し「参入障壁の低さが分散型の競争構造を生んでいる」
- リード文「参入障壁が低く新規参入が容易な一方、大日本印刷が緩やかな先行優位を築く。裾野が極めて広く、その中に緩やかな序列がある構造。」
- 根拠（カード）= HHI 0.0109・競争的／Entropy 8.48・高多様性／上位1社シェア 9.4%
- 締め文「包装・建材・印刷・化学など多様な産業が、自社事業のバイオマス化のために参入している帰結。」

#### A. レポートの論証を運ぶ（事実の刈り取り禁止）
- **各スライドは「1つの論点」を、主張→根拠（数値）→示唆（So what）の最小ロジックで語る**。`reports/report.typ` の該当章には既にこの連鎖があるので、それを凝縮する（数値は `data/*.json` で裏取り）。
- 箇条書きは「5語の事実断片」を並べない。**各注釈項目を「数値を含む主張＋含意」の完結した1〜2行の思考**にする。
- **悪い例（薄い・事実の刈り取り）** ❌
  - ・特許は2019年ピーク／・学術は増加中／・ニュースは2017年盛り上がり
- **良い例（レポートの論理を運ぶ）** ✅
  - ・特許は**2019年208件でピーク後に減少**だが、公開ラグ（1.5〜3年）を踏まえると衰退と断定できない
  - ・一方**学術論文は2025年593件まで加速継続** — 特許化フェーズと研究フェーズに明確な位相差
  - ・この位相差こそ、**蓄電・電磁波など研究先行のホワイトスペース**が残る根拠（後段で深掘り）

#### B. 適量の文字量（薄すぎ・詰め込みすぎの両方を避ける）
- コンテンツ面の本文（リード文＋注釈）は **合計おおむね 250〜400 字** を目安（**良質デッキ実測: 中央 333 字・平均 296 字**。下限で作ると薄くなるので 250 字を最低ラインとする）。**1枚に論点は1つ**（複数詰め込まない）。
- リード文＝結論1〜2行（40〜90字）。注釈＝**4〜6項目、各1〜2行（各40〜70字）**。長い段落をそのまま貼らない／逆に体言止めの単語だけにもしない。
- **図がある面でも本文を薄くしない**（「図があれば文字は少なくてよい」ではない）。リード文＋読み取り注釈 **4〜6点** を必ず添えて版面8割を埋める。図が無い面はポンチ絵（マトリクス/フロー/ツリー/ドーナツ/ピラミッド）＋注釈で埋める（§0.8）。
- ⚠️ **初回生成で薄くなりがち**（実テストで頻発）: `reports/report.typ` の該当章には主張→根拠（数値）→示唆の連鎖が既にあるので、それを 250〜400 字に**凝縮する**のであって、evidence の短い説明文を1〜2行貼って終わらせない。薄い面がコンテンツ面の3割を超えると Phase D `Check 16e` で **FAIL**。

#### C. 物語アークはレポートに従う（モジュール羅列にしない）
- スライドの章立ては **レポートの章順**（前提→エグゼクティブサマリー→環境→俯瞰→動態→競争→**クロス統合**→仮説検証→結論・提言→将来見通し）に沿わせる。
- とくに **クロスモジュール統合・仮説検証・結論/提言は、複数モジュールを束ねた「束ねスライド」** にする（1モジュール1枚の機械的割付では、レポートの最も価値ある統合的洞察が落ちる）。

#### D. 出所（出典）の書き方 — 自社モジュールを"出所"にしない
- ❌ `（出所）NEBULA ハイプサイクル分析` のように **分析モジュール名を出所として掲げない**（情報の出どころは分析機能ではなくデータ）。モジュール/ブランド機能名（Saturn V TELESCOPE・MEGA PULSE 等）は **本文・リード文側**で「〜分析によれば」と使う（terminology.md §2-B/2-C で本文使用可）。
- ✅ 特許データ由来の発見 → **`（出所）本分析の特許データセット（日本語公報 N件・期間）`**（必要なら「を基にAPOLLO作成」を付す）。学術/ニュース由来 → `（出所）学術論文データ` / `ニュースデータ`。
- ✅ Web調査由来の事実 → **実際の出所**（サイト名・URL・取得日。`reports/report.typ` の脚注／付録C「Web調査出所一覧」から転記）。
- 同じ出所文言を全スライドに機械的に貼らない。その面の根拠データに対応させる。
- ⚠️ **事業・市場ファクトを特許データセット出所にしない**: 「〇〇社が世界初採用」「商用プラント稼働」「市場 CAGR・億ドル」「プレスリリース」等は**特許データから導けない外部事実**。これらを載せる面は出所を**付録C（Web調査）等の実出所**にする（`（出所）本分析の特許データセット` のままにしない）。1面に特許由来＋Web由来が混在する場合は両方の出所を併記する。

#### E. 数値・固有名の一貫性（全スライド・本文で統一）
- 同一の数値（件数・%・年・CAGR・順位）は**デッキ全体で必ず同じ値**にする。例:「旭化成 91件」と「旭化成 92件」が別スライドに出るのは不可。**Phase D で確定した数値を1つに固定**し、本編レポート・別冊・PPTX で食い違わせない。
- 企業名・クラスタ名・テーマ名の表記も統一（略称/正式名を混在させない）。
- 数値は `data/*.json`・`reports/report.typ` の確定値を正とし、スライド作成時に**目分量で丸め直さない**。

#### F. 章扉・締めの体裁
- 章区切りの結論メッセージは「**結論：**」等のラベルを付けず、**地の文の一文**で書く（§0.8 A の「この章の結論」をそのまま宣言調で）。
- 締め・フッターのブランド表記は **「APOLLO」に統一**（`APOLLO CAPCOM` は不可）。裏表紙は v6 定型（レポートタイトル＋「APOLLO Patent Analytics Platform」＋日付）で、**Thank You・タグライン等の飾り文言は置かない**。

#### G. 図形の充填 — 「でかい四角に少しの文字」を作らない（スカスカ撲滅）
- カード／象限／ステップ等の**箱には、見出し＋2〜4行の具体的な説明（数値を含む）**を入れる。**大きな箱に1行だけ**置いて下半分が空く状態を作らない。
- 文字が少ししか無いなら、**箱を小さくする・箱を統合する・要素数を減らす**（埋められない数の箱を並べない。例: 内容が3つなら3カードにする）。
- **箱内テキストは上下中央寄せ**（`text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE`）。短文が上端に張り付いて下が空く見え方を防ぐ（`add_cards_slide` / `add_matrix_2x2_slide` 等は実装済み）。
- 1スライドの箱数の目安: カード3〜4・象限4・ステップ3〜5・バンド2〜4。**箱が多いほど各箱が薄くなる**ので、内容に対して箱を増やしすぎない。

---

## Section 1: デザインシステム

### 設計原則

#### 原則1: Fill the slide — 余白禁止
コンテンツ領域はタイトル下端からフッター上端まで全面使用する。
リード文がある場合はコンテンツ領域を下にシフトし、残り領域を全てコンテンツで埋める。空きスペースは許可しない。

#### 原則2: 動的カードサイジング
- KPIカード: 4個以下 = 1行、5-8個 = 2行。幅 = (利用可能幅 - ギャップ合計) / 列数
- プロセスステップ: 2個以下 = 大ボックス、3個 = 中、4個以上 = コンパクト。フォントも連動縮小
- テーブル行: 行高 = (利用可能高さ) / 行数（最小0.35、最大0.55）

#### 原則3: 全スライド共通構造（v6）
表紙・章扉・裏表紙以外の全スライドに以下を適用:
1. タイトル（22pt Bold INK・結論を言い切る文）＋ **ACCENT 短下線**（幅 0.55in × 2.25pt。全幅下線は廃止）
2. リード文（12.5pt Medium SUB・**プレーン文**。背景箱・■は廃止。省略可）
3. コンテンツ領域（白角丸カード等で残り全スペースを使い切る）
4. 出所行（8.5pt Light MUTED・`（出所）...`）
5. フッター（**ヘアライン**＋左 `APOLLO｜レポートタイトル｜日付` 8pt Light・右 頁番号 9pt Light）

#### 原則4: タイトル＝結論（新聞見出し方式・見出し規定の唯一の正）
タイトルは結論そのもの。単なるラベルではない。**主張見出しの規定は本原則に一本化する**（§0.9-A0-2 も本原則に従う）。

| NG ラベル型 | OK 結論型 |
|------------|----------|
| 「クラスタ分析結果」 | 「**上位5クラスタが全体の58%を占有。技術集中化が加速**」 |
| 「出願動向」 | 「**出願件数はCAGR 20%で成長—2022年ピーク後は選択と集中へ**」 |
| 「競合分析」 | 「**A社がシェア首位も、B社がSiC領域で急追。3年以内に逆転も**」 |

タイトルは結論を1文で言い切る。数値を必ず含める。**波ダッシュ「～」は使わない**（補足が要れば全角ダッシュ「—」か句点で短い2文に分ける）。

#### 原則5: 可視化ファースト
| 形式 | 使用割合 | 用途 |
|------|---------|------|
| **チャート+注釈** | **50%以上** | チャート主体 + 注釈カードレール |
| **ポンチ絵付き** | **20%以上** | カード/バンド/プロセス/ピラミッド/比較等 |
| **テーブル+注釈** | 10-15% | 数値比較テーブル + 読み取り/決め手 |
| ナラティブ（テキスト主体） | **10%以下** | エグゼクティブサマリー + 結論のみ |

### デザイントークン（v6 新設・カラーの正）

**2層サーフェス**が土台: スライド地は BG（クールオフホワイト）、コンテンツは CARD（白・角丸 0.07in・CARD_LINE のヘアライン枠）に載せる。数値・本文は INK（Navy インク）主体で、**色は「意味」にだけ使う**。

#### 基本サーフェス・文字色

| トークン | 値 | 用途 |
|---------|-----|------|
| `BG` | `#F4F6F9` | スライド地（クールオフホワイト） |
| `CARD` | `#FFFFFF` | カード地（角丸 0.07in） |
| `CARD_LINE` | `#E3E8F1` | カード枠（ヘアライン 0.75pt）・フッター線 |
| `INK` | `#1B2A4A` | 主文字色（タイトル・KPI数値・バンド見出し） |
| `DEEP` | `#152238` | 濃紺地（**表紙・裏表紙・結論ピル・テーブルヘッダー**） |
| `SUB` | `#4F5B70` | 副文字色（リード文・注釈本文） |
| `MUTED` | `#8792A3` | 出所・フッター・キャプション |
| `GHOST` | `#E9EEF6` | 章扉ゴースト数字 |
| `ACCENT` | `#2A78D6` | ブランドアクセント（短下線・表紙ピル・下端帯。CAT 青と同一） |
| `WHITE` | `#FFFFFF` | 濃色地の上の文字 |
| `SKY` | `#8FB8E8` | 濃紺地の上のアクセント文字（表紙サブタイトル） |
| `SUB_DK` | `#B9C6DC` | 濃紺地の上の副文字 |
| `MUT_DK` | `#7687A5` | 濃紺地の上の弱文字 |

#### カテゴリ4色（CAT）と派生（TINT / CHIP）

**色覚多様性（CVD）検証済み**（最悪隣接ΔE 73.6）。**色は意味（カテゴリ識別・判定）にだけ使い、アクセントバー・チップ・ティント地に載せる。文字に CAT 彩色を載せない**（識別は常にラベル併記）。

| キー | `CAT`（バー・基準色） | `TINT`（淡色帯地） | `CHIP`（白文字用濃色変種） |
|------|---------------------|-------------------|---------------------------|
| blue | `#2A78D6` | `#E9F1FB` | `#1F5FB0` |
| teal | `#1BAF7A` | `#E5F6EF` | `#12805A` |
| violet | `#4A3AA7` | `#EDEAF7` | `#4A3AA7` |
| amber | `#EDA100` | `#FCF3DD` | `#B26A00` |

- **CAT**: カードの左バー（3.6pt）／上バー（3.2pt・角丸を避けて左右 0.1in 内側）・軸/凡例マーカーに使う。
- **TINT**: バンド地・注意/決め手ボックス地・カードヘッダ地。TINT 地の文字は INK/SUB（白文字不可）。
- **CHIP**: 白文字を載せるラベルピル（バンドのチップ等）。**白文字を載せてよいのは CHIP と DEEP のみ**（CAT・TINT には白文字を載せない）。

#### 判定色（テーブル・仮説検証）

| トークン | 値 | 用途 |
|---------|-----|------|
| `GOOD_TX` | `#1E7B34` | 判定「✓ 支持」の文字色 |
| `PART_TX` | `#B26A00` | 判定「△ 部分支持」・「注意」「決め手」ラベルの文字色 |

#### レイアウト定数

| 定数 | 値 | 用途 |
|------|-----|------|
| 左右マージン `MX` | 0.55in | 全スライド共通 |
| コンテンツ幅 `CW` | 13.333 − 2×MX in | 16:9（13.333×7.5in） |
| カード角丸 | 半径 0.07in（絶対値） | 白カード共通 |
| 左アクセントバー | 3.6pt（Emu 45720） | カード/バンド/コールアウトの左端 |
| 上アクセントバー | 3.2pt（Emu 41148）・左右 0.1in 内側 | KPIカード上辺（角丸を避ける） |
| 短下線 | 0.55in × 2.25pt ACCENT | タイトル直下 |
| フッター線 | ヘアライン（CARD_LINE）y≈7.02 | フッターの上 |
| 出所行 | y≈6.72 | フッターの上 |
| 下端帯 | 全幅 × 0.16in ACCENT | **表紙・裏表紙のみ** |

> 旧 v5 定数（`NAVY` / `DARK_GRAY` / `RED_ACCENT` 等）は互換のため `apollo_slides.py` に残る場合があるが、**新規デザインは本節のトークンを正**とする（棄却/警告など v6 に対応トークンが無い意味色のみ旧定数を継続使用可）。

### フォント

全スライドで **Noto Sans JP（ゴシック）を統一使用**する（日本語・欧文とも。明朝/セリフ・等幅は使わない）。**ウェイトを多段に使い分けて編集的な階層を作る**（`_apply_font(run, weight=...)` / `set_text(..., weight=...)` / `add_rich_runs(..., weight=...)` で指定）。ウェイトは名前付きファミリ（Light/Regular/Medium/SemiBold/Black）＋ **Bold（Noto Sans JP ＋ bold フラグ。Black より一段上品な太さ）**。未インストール環境では近いウェイトに自動フォールバックする。全ランに `lang="ja-JP"`＋`a:ea` 明示（中華風グリフの構造的防止）。

**ウェイト階層（v6・厳守）**: **タイトル・KPI数値・バンド見出し＝Bold**（**Black は章扉ゴースト数字のみ**）／**カード見出し＝SemiBold**／**リード文・チップ＝Medium**／**本文・注釈＝Regular**（強調語のみ Bold）／**出所・フッター・キャプション＝Light**。同一行で複数ウェイトを混ぜすぎない。

| 要素 | サイズ | ウェイト | 色 |
|------|-------|---------|-----|
| 表紙タイトル | **40pt** | **Bold** | WHITE |
| 表紙サブタイトル | 14pt | SemiBold | SKY |
| 表紙リード段落 | 11.5pt | Regular（行間1.45） | SUB_DK |
| 表紙チップ（APOLLO REPORT） | 9.5pt | Medium（字間広め） | WHITE on ACCENT |
| 章扉ゴースト数字 | **230pt** | **Black（唯一の使用箇所）** | GHOST |
| 章扉「SECTION NN」 | 10pt | Medium（字間広め） | MUTED |
| 章扉タイトル | **30pt** | **Bold** | INK |
| 章扉サブ行（章の結論） | 13pt | Medium | SUB |
| スライドタイトル（結論文） | **22pt**（長文時自動縮小） | **Bold** | INK |
| アイブロウ | 10pt | Medium（字間広め） | MUTED |
| リード文 | 12.5pt | Medium | SUB |
| KPI 大数値 | **30pt**（表紙内は21pt） | **Bold** | INK |
| KPI ラベル | 11pt | SemiBold | SUB |
| カード見出し（注釈/読み取り） | 10〜11pt | SemiBold | INK |
| バンド見出し | **16pt** | **Bold** | INK |
| チップ（バンドラベル） | 9.5pt | Medium（字間+60） | WHITE on CHIP |
| 本文・注釈 | 9〜10pt | Regular（強調のみ Bold・行間1.25〜1.4） | SUB / INK |
| 締め文 | 10〜10.5pt | Medium または Regular | SUB |
| 結論ピル | 12.5pt | SemiBold | WHITE on DEEP |
| 表ヘッダー | 10.5pt | SemiBold | WHITE on DEEP |
| 表本文 | 9.5pt | Regular（先頭列・判定列は SemiBold） | INK / SUB / 判定色 |
| キャプション | 9pt | Light | MUTED |
| 出所 | 8.5pt | Light | MUTED |
| フッター | 8pt（頁番号 9pt） | Light | MUTED |

> これらのフォント定数（`FONT_FAMILY` / `WEIGHT_FAMILY` / `JA_FONT` / `LATIN_FONT`）は **`apollo_slides.py` に定義済み**（上表は参照用の値）。

### テンプレートとセットアップ

テンプレート PPTX（`apollo_template.pptx`）・スナップショットフォルダのパス定数（`TEMPLATE` / `SNAP`）も **`apollo_slides.py` に定義済み**。ただし `Presentation` の生成・`blank` レイアウトの取得・`SNAP` の実フォルダ上書きは **呼び出しスクリプト側の責務**（→ Section 2「import 運用」）。

---

## Section 2: import 運用 + コア／描画補助ヘルパー・カタログ

ヘルパー本体（フォント・色・レイアウトの実装）は **`capcom_schema/templates/apollo_slides.py`** に抽出済み。生成スクリプトは**コードをコピーせず import して使う**。Section 2 はコア／描画補助ヘルパーのカタログ、Section 3 はスライドタイプのカタログ（**コード本体は載せない**。関数シグネチャ＋用途＋主張骨格メモ＋使い分けルールのみ）。

### import の使い方（生成スクリプトの冒頭）

```python
import sys
sys.path.insert(0, "capcom_schema/templates")   # apollo_slides.py のあるフォルダ
from apollo_slides import *                       # 全 add_*_slide・コア関数・色・フォント定数が入る

from pptx import Presentation
import os

# prs と blank は呼び出しスクリプトが自前で用意して各 add_*_slide に渡す
prs = Presentation(TEMPLATE)          # TEMPLATE は apollo_slides に定義済み（apollo_template.pptx）
blank = prs.slide_layouts[6]          # ブランクレイアウト（全 add_*_slide に渡す引数）

# スナップショット実フォルダを指す（相対パス画像の解決基点を上書き）
import apollo_slides
apollo_slides.SNAP = os.path.join(session_dir, "snapshots")

# 以降、各ヘルパーを呼んでスライドを積む（戻り値連鎖は §0.5）
add_title_slide(prs, "...", "...", "2026-06-27", blank)
# ...
prs.save("reports/apollo_report_YYYYMMDD.pptx")
```

- `prs`（`Presentation`）と `blank`（`prs.slide_layouts[6]`）は **呼び出しスクリプトが作って各 `add_*_slide` の `prs` / `blank` 引数に渡す**（モジュールは import 時に `Presentation` を生成しない）。
- `SNAP` は相対パス画像（`add_chart_text_slide` 等の `image_path`）の解決基点。CAPCOM セッションの `snapshots/` 実フォルダに上書きする。
- 色（`INK` / `CAT` / `TINT` / `CHIP` 等）・フォント（`FONT_FAMILY` 等）・`TEMPLATE` も `from apollo_slides import *` で利用可能。

### コアヘルパー（タイトル/リード文/フォント）

これらは全スライドの土台。`add_title_shape` → `add_sub_message` の**戻り値を連鎖**させてレイアウトを組む（§0.5）。

- **`_apply_font(run, weight=None)`** — run にデュアルフォント（欧文＋日本語）＋ `lang="ja-JP"` ＋名前付きウェイト（light/regular/medium/semibold/bold/black）を設定。全テキストの土台で、直接呼ぶことは少ないが品質の核。
- **`_apply_kinsoku(paragraph)`** — 段落に日本語禁則処理（行頭・行末禁則）を設定。
- **`add_rich_runs(paragraph, text, base_size=Pt(14), base_color=SUB, bold_color=None, force_bold=False, line_spacing=1.4, weight=None)`** — `**太字**` マーカーを解析しつつデュアルフォント＋禁則＋行間＋ウェイトを適用。注釈・本文の整形に使う。
- **`set_text(p, text, size, color, bold=False, line_spacing=None, weight=None)`** — 単純テキスト1本を整形（デュアルフォント＋禁則＋ウェイト）。ラベル・見出し・1行テキストに。
- **`add_title_shape(slide, text, x=0.5, y=0.15, w=12.3, eyebrow=None)`** → リード文開始 y を返す。スライドタイトル（**22pt Bold INK ＋ ACCENT 短下線 0.55in×2.25pt**。全幅下線は廃止）。タイトル長でフォントサイズと高さを自動調整。
  - 主張骨格メモ（§0.9-A0）: `text` = **主張見出し**（**原則4に従い結論を言い切る文**。ラベル・短い名詞句にしない・数値を含める・「～」は使わず必要なら「—」/句点で2文）。`eyebrow` = **アイブロウ**（章/モジュール名。例 `"NEBULA / 環境分析"`。10pt Medium・字間広め・ミュート色・ゴシック統一）。
- **`add_sub_message(slide, message, x=0.5, y=None, w=12.3)`** → コンテンツ開始 y を返す。**リード文をプレーン文（12.5pt Medium SUB）で描画**する。v5 の KEY_MSG_BG 背景箱・左バー・■は**廃止**。`y` は `add_title_shape` の戻り値を渡す。
  - 主張骨格メモ: `message` = **リード文**（核心主張の完結した一文・数値込み・40〜90字・2行以内）。⚠️ `message` に■を含めない（v6 は■を描画しない。含まれていても除去される）。

### 描画補助（フッター/画像/注釈カード/コールアウト/出典）

- **`add_bottom_bar_and_footer(slide, page_num=None)`** — 全コンテンツスライド共通の**フッター新形式**: ヘアライン（CARD_LINE・y≈7.02）＋左 `APOLLO｜レポートタイトル｜日付`（8pt Light MUTED）＋右 頁番号（9pt Light MUTED・2桁ゼロ埋め）。レポートタイトル・日付はモジュール側の設定で与える（実装は `apollo_slides.py` 参照）。**全コンテンツスライドの最後に必ず呼ぶ**。表紙・章扉・裏表紙では呼ばない。
- **`fit_image(slide, image_path, max_x, max_y, max_w, max_h)`** — 画像をアスペクト比保持で指定領域内に中央配置。存在しなければ `None`。v6 では**画像は白カードに載せ**（`_card` の上に pad≈0.12in で中央配置）、カード下端にキャプション（9pt Light MUTED）を添える。
- **`add_annotation_cards(slide, cards, x, y, w, card_h=None)`** — **注釈カードレール（v6 新設）**。図の右（または左）レールに白カード＋CAT 左バー 3.6pt＋見出し（10pt SemiBold INK）＋本文（9pt Regular SUB）を縦積みする。`cards=[{"header","body","color"}, ...]`（3〜4枚）。
  - **推奨4部構成: 「図の見方」(blue) →「読み取り」(teal) →「別の見方」(amber) →「示唆」(violet)**。「別の見方」で別解釈とそれを退ける決め手を1枚添えること — これは**レポート本文の構造化分析（結論の検証: 別解釈＋決め手）とスライド側で対応**し、図の読みを一方向に流さないための仕掛け。
- **`add_callout_box(slide, label, body, x, y, w, h, color="amber")`** — **注意・決め手ボックス(v6 新設）**。TINT 地（角丸）＋CAT 左バー 3.6pt＋ラベル（11pt SemiBold・「注意」「決め手」等。amber 系は PART_TX 色）＋本文（9.5〜10.5pt Regular INK）。
  - 使いどころ: **公開遅延の注意**（「直近2年の件数は公開ラグで過小。減少と読まない」）、**表・仮説検証の決め手の明示**（「主軸は特許・市場・製品発表が同方向。MRAM は外部裏付けが薄く定点観測対象と判定」）、前提と見直しのサイン等。
- **`add_source_label(slide, source_text, x=0.5, y=6.55, w=12.3)`** — `（出所）...` ラベル（8.5pt Light MUTED）。出所ルールは §0.9-D（モジュール名を出所にしない）。
- **`add_annotation_block(slide, bullets, x, y, w, h, font_size=14, has_border=False, bg_color=None)`** — 箇条注釈ブロック（チャート横の分析テキスト）。各 bullet は「数値を含む完結文」（§0.9-A0/A。5語断片を並べない）。v6 では図＋注釈面は `add_annotation_cards` を優先し、本関数はカード化しない小さな注釈に使う。
- **`add_chart_label(slide, text, x, y, w=3.0, size=14, color=INK)`** — チャート小見出し（グラフ上の分類ラベル）。
- **`add_chart_callout(slide, text, x, y, w=2.5, arrow_to_x=None, arrow_to_y=None, bg_color=None, font_size=12, border_color=None)`** — チャート上に吹き出し注釈をオーバーレイ（任意で対象点へ矢印）。
- **`add_highlight_circle(slide, x, y, w=0.5, h=0.5, color=None)`** — チャート上のハイライト丸囲み。
- **`chip(slide, x, y, text, color_key="blue", w=1.15, h=0.30, size=9.5)`** — ラベルピル部品。CHIP 濃色地（角丸 0.05in）＋白 Medium 文字（字間+60）。バンドのラベル・凡例チップに。

> ⚠️ 作図共通の禁止事項: **コネクタ（`p:cxnSp`）不使用**（矢印は CHEVRON/ARROW オートシェイプ）・**GroupShape 不使用**（部品は個別シェイプで直接配置）・**CAT/TINT 地に白文字を載せない**（白文字は CHIP/DEEP のみ）。

---

## Section 3: スライドタイプ・カタログ（16種＋補助）

各 `add_*_slide(prs, ..., blank, ...)` は1関数=1スライドを `prs` に追加して返す。`prs` と `blank` は import 運用（Section 2）で用意したものを渡す。**コード本体は `apollo_slides.py` 参照**。各エントリは「シグネチャ＋用途＋（該当すれば）主張骨格メモ・使い分けルール」のみ。

### 構造スライド（表紙・章扉・裏表紙。フッターを呼ばない）

- **`add_title_slide(prs, title, subtitle, date, blank)`** — **表紙（v6: DEEP 濃紺基調・裏表紙と同系統）**。構成: 左上に ACCENT ピル「APOLLO REPORT」（9.5pt White Medium・字間広め）→ 40pt Bold White タイトル → ACCENT 小バー → サブタイトル（14pt SKY SemiBold・Mission Objective の要約）→ リード段落（11.5pt SUB_DK・分析の読み方を2〜3行）→ **KPI カード3枚**（白カード・枠線なし＋CAT 左バー・値 21pt Bold INK）→ 母集団注記（8.5pt MUT_DK）。右半分に**主図版の白カード**（俯瞰図等＋キャプション）。最下端に全幅 ACCENT 帯（0.16in）。飾りだけの空白面にしない（§0.8 A）。
- **`add_section_slide(prs, section_num, title, blank, subtitle=None)`** — **章扉（v6: ライト章扉・BG 地）**。濃紺背景は廃止。右にゴースト数字（**230pt GHOST 色 Black・右寄せ**。Black の唯一の使用箇所）、左に「SECTION NN」（10pt MUTED Medium・字間広め）→ ACCENT 小バー → 章タイトル（30pt Bold INK）→ サブ行（13pt Medium SUB）。**サブ行には「この章の結論メッセージ」（20-40字・地の文）を必ず与える**。章扉はデッキの15%以下（§0.8 A／§0.9-F）。
- **`add_closing_slide(prs, report_title, blank)`** — **裏表紙（v6: DEEP 濃紺・"Thank You" 廃止）**。構成: 中央に ACCENT 小バー → レポートタイトル（28pt Bold White・中央揃え）→「APOLLO Patent Analytics Platform」（11pt Medium・字間広め・濃紺地用ミュート色）→ 日付（10pt Light）。最下端に全幅 ACCENT 帯（0.16in）。**Thank You・タグライン等の飾り文言は置かない**。次アクションは直前の `add_action_items_slide` が担う（§0.8 A）。

### 主力スライド（チャート+注釈・KPI）

- **`add_chart_text_slide(prs, title, sub_message, image_path, annotations, blank, caption=None, chart_label=None, text_side="right", chart_ratio=0.60, source=None, page_num=None, eyebrow=None)`** — **主力タイプ（デッキの50%以上）**。左に白カードに載せたチャート画像（`chart_ratio` で幅比 0.55-0.65）、右（または左）に**注釈カードレール**（`add_annotation_cards` 体裁: 白カード＋CAT 左バー＋見出し SemiBold＋本文）。`image_path` は `SNAP` 基点の相対パス可。
  - 主張骨格メモ（§0.9-A0・厳守）: `title` = **主張見出し（結論を言い切る文）**／`sub_message` = **リード文**（核心主張の完結文・数値込み）／`annotations` = **根拠の完結文**（5語断片でなく各1〜2行・4〜6項目）。右レールは **「図の見方/読み取り/別の見方/示唆」の4部構成を推奨**（「別の見方」＝別解釈と決め手。レポートの構造化分析と対応）。**最後の1項目（示唆）は必ず「締め文（So What）」**＝地の文の一文。`eyebrow` でアイブロウを添える。
- **`add_kpi_slide(prs, title, sub_message, kpis, blank, source=None, page_num=None)`** — **主力タイプ**。KPI ダッシュボード。`kpis=[{"label","value","unit","trend"}, ...]`。4個以下=1行／5-8個=2行に自動配置。**v6 カード体裁: 白角丸カード（CARD_LINE 枠）＋上辺 3.2pt CAT バー（角丸を避け左右 0.1in 内側）＋大数値（30pt Bold INK）＋ラベル（11pt SemiBold SUB）＋補足（9pt Light MUTED）**。数値は Black でなく Bold・彩色しない（色はバーだけが担う）。エグゼクティブサマリーに KPI 3〜4 個（§0.7）。KPI 行の下は「読み取り」カード（箇条3点＋締め文）＋`add_callout_box`（注意）で版面を埋める。

### ポンチ絵スライド（カード/マトリクス/フロー/ドーナツ/ツリー 等）

- **`add_cards_slide(prs, title, sub_message, cards, blank, source=None, page_num=None)`** — 3〜4枚のカード横並び。`cards=[{"header","body","color"}, ...]`（`body` は文字列 or 箇条リスト）。v6 体裁: 白角丸カード＋CAT 左バー＋見出し SemiBold INK＋本文 Regular SUB。箱内テキストは上下中央寄せ実装済み。**大箱に1行だけにしない**＝内容が3つなら3カード（§0.9-G）。
- **`add_matrix_2x2_slide(prs, title, sub_message, x_axis, y_axis, quadrants, blank, source=None, page_num=None)`** — **軸付き2×2マトリクス（推奨）**。`x_axis`/`y_axis`＝`{"label","low","high"}`、`quadrants`＝[左上,右上,左下,右下] 各 `{"label","desc","color"}`。象限地は TINT・ラベルは INK（白文字を載せない）。
  - 使い分けルール（§0.8 C・厳守）: ただの4分割矩形にしない。**縦横の軸（矢印＋ラベル「← … →」＋低/高）と各象限の意味ラベル・説明を必ず描く**（本関数が軸矢印・低/高・象限ラベルを自動描画）。象限は厳密に4要素。コネクタは使わない＝RIGHT_ARROW/UP_ARROW オートシェイプで軸を描く。成長×活動量・重要度×緊急度・BCG 等に。
- **`add_arrow_flow_slide(prs, title, sub_message, steps, blank, source=None, page_num=None)`** — 横向き矢羽根フロー（プロセス/因果）。`steps=[{"title","desc"}, ...]`（3〜6個）。先頭=PENTAGON、以降=CHEVRON。
  - 使い分けルール（§0.8 C・厳守）: **コネクタ（`p:cxnSp`）禁止＝矢羽根（CHEVRON/PENTAGON）で流れを描く**（コネクタは python-pptx で破損）。`add_process_slide`（縦STEP）の横版。
- **`add_donut_slide(prs, title, sub_message, segments, blank, center_label=None, source=None, page_num=None)`** — ドーナツ図（構成比）。`segments=[{"label","value","color"}, ...]`（3〜4推奨・CAT 4色）、`center_label` 中央の大数値（任意）。右に色見本付き凡例（ラベル＋%）。
  - 使い分けルール（§0.8 C）: 構成比は**色付き矩形の羅列で代用せず本関数（`BLOCK_ARC`）を使う**。⚠️ 実装上の罠（温存済み・不変）: python-pptx は blockArc 角度を「度×100000」で格納するため、実角度°→adjustment値は **×0.6**（=60000/100000）換算が必須（怠ると扇形が約1.67倍に拡大し破綻）。内径比（`adj3`）は 0〜1 比率のまま。
- **`add_issue_tree_slide(prs, title, sub_message, root, branches, blank, source=None, page_num=None)`** — Issue Tree／ロジックツリー（2階層）。左に論点（`root`＝`{"title","desc"}`）、右に分解枝（`branches`＝[{"title","desc"}, ...] 2〜5）。
  - 使い分けルール（§0.8 C）: 「なぜ？／何が要因か？」の分解に。**枝線・分岐は細い矩形＋RIGHT_ARROW オートシェイプで描く（コネクタ不使用）**。
- **`add_process_slide(prs, title, sub_message, steps, blank, source=None, page_num=None)`** — 縦STEPプロセスフロー。`steps=[{"title","desc"}, ...]`。2個以下=大ボックス/3個=中/4個以上=コンパクト（フォント連動）。戦略提言のポンチ絵化に（§0.7）。
- **`add_stepup_slide(prs, title, sub_message, phases, blank, source=None, page_num=None)`** — 階段型ロードマップ（左→右で棒が高くなる）。`phases=[{"header","body","color"}, ...]`（3〜4段推奨）。短期→中期→長期に。戦略提言のポンチ絵化に。
- **`add_compare_slide(prs, title, sub_message, left_title, left_items, right_title, right_items, blank, left_color=None, right_color=None, source=None, page_num=None)`** — 左右比較（中央"VS"＋区切り線）。`left_items`/`right_items`＝各3〜5項目。色は CAT から（例: blue vs amber）。特許 vs 学術、A社 vs B社に。
- **`add_progress_bar_slide(prs, title, sub_message, items, blank, source=None, page_num=None)`** — 水平プログレスバー。`items=[{"label","value","max_value","color"}, ...]`。CAGR 比較・シェア表示に。
- **`add_triangle_slide(prs, title, sub_message, elements, blank, source=None, page_num=None)`** — 3要素トライアングル関係図（上1＋下2＋関係矢印）。`elements=[{"title","body","color"}, ...]`（3要素）。技術-市場-政策、3者競合に。
- **`add_pyramid_slide(prs, title, sub_message, levels, blank, source=None, page_num=None)`** — ピラミッド（台形積み重ね）。`levels=[{"title","detail"}, ...]`（上→下の順）。技術階層（基盤→応用→萌芽）、ノイズ分析に。
- **`add_timeline_slide(prs, title, sub_message, events, blank, source=None, page_num=None)`** — 水平タイムライン（マーカー＋年＋ラベル上下交互）。`events=[{"year","title","color"}, ...]`。政策イベント・技術マイルストーンに。

### データスライド（表・仮説検証）

- **`add_table_slide(prs, title, sub_message, headers, rows, blank, col_widths=None, highlight_rows=None, annotations=None, source=None, page_num=None)`** — 表スライド。**v6 体裁: ヘッダー行＝DEEP 濃紺地＋白 SemiBold（10.5pt）、本文行＝CARD/BG のゼブラ（9.5pt・先頭列 SemiBold INK）、判定列＝`GOOD_TX`（✓ 支持）/`PART_TX`（△ 部分支持）の SemiBold 彩色文字**。`annotations` を渡すと表横に注釈枠。行高は残余スペースに動的フィット。
  - 使い分けルール（§0.8 B・厳守）: **表を置きっぱなしにしない**。表の上にリード文（結論）1行、周辺に「この表から何を読むか」を注釈2〜3点。**判定の分かれ目は表の下の `add_callout_box(label="決め手", ...)` で言語化**する。とくに権利化率表・仮説検証表・将来見通し表は示唆を言語化する。
- **`add_hypothesis_slide(prs, title, sub_message, hypotheses, blank, source=None, page_num=None)`** — 仮説検証テーブル（ID／仮説／判定／エビデンス）。`hypotheses=[{"id","hypothesis","verdict","evidence"}, ...]`。`verdict`＝`"confirmed"→✓ 支持（GOOD_TX）` / `"partially"→△ 部分支持（PART_TX）` / `"rejected"→✕ 棄却（RED_ACCENT 互換色）`。各仮説に判定と根拠を必ず（§0.7）。クロス統合の束ねスライドに。判定が割れた仮説は `add_callout_box` の「決め手」で判定理由を明示。

### 章構成スライド（主張骨格 §0.9-A0 を体現）

エグゼクティブサマリー・クロス統合・結論/提言の各パートで、断片箇条書きでなく**1枚＝1主張**を完結文で論じる面を作るのに使う。いずれも `add_title_shape(slide, title, eyebrow=…)` でアイブロウ＋短下線、`add_bottom_bar_and_footer()` でフッターを描く。矢印は MSO_SHAPE オートシェイプ（コネクタ `p:cxnSp` 不使用）。

- **`add_bands_slide(prs, title, lead, bands, blank, conclusion=None, closing=None, eyebrow=None, source=None, page_num=None)`** — **結論バンド面（v6 新設）**。全幅バンド（**TINT 淡色地・角丸＋CAT 左バー 3.6pt＋CHIP チップ（白 Medium・層ラベル）＋バンド見出し（16pt Bold INK）＋補足1行（9.5pt Regular SUB）**）を 2〜4 本縦積みし、下部に任意の**結論ピル**（`conclusion`: DEEP 濃紺・角丸・白 12.5pt SemiBold・中央揃え）と**締め段落**（`closing`: 10.5pt Regular SUB・2〜3行）を置く。`bands=[{"chip","heading","note","color"}, ...]`。
  - 使いどころ: **結論・提言・層構造の提示**（例: 短中期の主軸／主軸の拡張／長期オプションの3層）。各バンド見出しは言い切りの文または技術群名＋数値、補足に件数・構成比の根拠を必ず入れる。白文字はチップ（CHIP）と結論ピル（DEEP）のみ — バンド地に白文字を載せない。
- **`add_shift_slide(prs, title, lead, past, present, closing, blank, eyebrow=None, source=None, page_num=None)`** — **重心移動（PAST → PRESENT）**。左カード「PAST ・ 過去の主役」、中央に右向き矢印（オートシェイプ）、右カード「PRESENT ・ 現在の重点」を同サイズで並べ、下部に頑健性の締め文を置く。`past`/`present`＝`{"label","heading","desc"}`（label=小ミュート／heading=Bold INK／desc=Regular。PRESENT 側を ACCENT 帯で強調）。`lead`＝リード文、`closing`＝締め文（地の文一文）。
  - 主張骨格メモ（§0.9-A0）: アイブロウ→主張見出し→リード文→根拠（PAST/PRESENT 2枚）→締め文（So What）。**エグゼクティブサマリー**の主役交代・重心移動を1枚で言い切る面に。
- **`add_convergence_slide(prs, title, methods, conclusion, blank, eyebrow=None, source=None, page_num=None)`** — **クロス統合（N手法 → 1つの頑健な結論へ収束）**。左に N 個の手法行（`methods=[{"method","finding"}, ...]` 3〜5件。手法名＝Medium INK／発見＝Regular）、各行から細い矢印が右の大きな結論ボックス（`conclusion={"headline","detail"}`・ACCENT 帯＋TINT 背景・headline=Bold／detail=Regular）へ収束する。
  - 主張骨格メモ（§0.9-A0・§0.9-C）: 複数手法を束ねた「束ねスライド」。各手法の発見＝根拠、右の結論箱＝主張見出し＋So What。**クロスモジュール統合**で「N手法が同じ結論」を示す面に。矢印は `_add_line()`＋RIGHT_ARROW（コネクタ不使用）。
- **`add_priority_actions_slide(prs, title, actions, blank, eyebrow=None, source=None, page_num=None)`** — **優先度別アクション**。各行に [優先度バッジ 高/中/低（色ピル）] ＋ 見出し ＋ 詳細1文 ＋ [期間ピル]。`actions=[{"priority":"高"/"中"/"低","title","detail","timeframe"}, ...]`。優先度色＝高(RED_ACCENT 互換色)/中(PART_TX)/低(MUTED)、期間は枠線の小ピル。
  - 使い分け: 既存 **`add_recommendation_slide`**（優先度バー＋期間＋詳細）のバッジ＋期間ピル版。**優先度＋期間＋詳細だけで足りるなら `add_recommendation_slide` でも可**。バッジ/ピル体裁で結論・提言部を整えたいときに本関数を使う。
- **`add_action_items_slide(prs, title, items, blank, eyebrow=None, brand_line=None, source=None, page_num=None)`** — **アクションアイテム（☐ チェックリスト）**。`items=[str, ...]`（各「完結したアクション文」。☐ は関数が付与）を縦に並べ、下部に任意の `brand_line`（例「APOLLO ・ 特許ランドスケープ分析」）。締め・結論寄りの体裁。
  - 使い分け: **結論・提言部の末尾／裏表紙の直前**に、合意済みの次アクションをチェックリストで列挙する面に（v6 裏表紙は Thank You なしの定型のため、次アクションは必ずこの面が担う）。

### 補助スライド

- **`add_toc_slide(prs, title, items, blank, page_num=None)`** — 目次（ゼブラストライプ）。`items=[{"num","title","page"}, ...]`。
- **`add_dual_panel_slide(prs, title, sub_message, left_label, left_image, left_caption, right_label, right_image, right_caption, left_bullets=None, right_bullets=None, blank=None, source=None, page_num=None)`** — 2カラムチャート比較（2つの可視化を並列・各カラム白カード）。各カラムに画像＋キャプション＋任意の注釈。
- **`add_narrative_slide(prs, title, sub_message, paragraphs, blank, source=None, page_num=None)`** — テキスト主体。**エグゼクティブサマリーと結論にのみ使用し、全スライドの10%以下に制限**（§0.7）。
- **`add_image_slide(prs, title, sub_message, image_path, blank, caption=None, chart_label=None, source=None, page_num=None)`** — チャート全画面（画像が主役・白カード載せ）。貼った図には読み取り注釈を併記（§0.8 C）。
- **`add_recommendation_slide(prs, title, sub_message, recommendations, blank, source=None, page_num=None)`** — 推奨アクション（優先度バー付き）。`recommendations=[{"priority","title","timeframe","desc"}, ...]`。`priority`＝高(赤)/中(黄)/低(緑)。
- **`add_matrix_slide(prs, title, sub_message, quadrants, blank, x_label="→ 成長率", y_label="↑ 規模", source=None, page_num=None)`** — **旧式2×2（軸ラベルのみ）**。`quadrants={"TL","TR","BL","BR"}`。新規面は軸矢印付きの **`add_matrix_2x2_slide()` を優先**（こちらは軸が矢印でなくテキストラベルのみ）。

### 主要スライドタイプの使い分け早見

| 関数 | 分類 | 使いどころ |
|------|------|-----------|
| `add_chart_text_slide()` | **主力** | チャート＋注釈カードレール（デッキの50%以上）。主張骨格を載せる中核 |
| `add_kpi_slide()` | **主力** | KPI ダッシュボード（白カード＋上辺 CAT バーの新体裁） |
| `add_bands_slide()` | 章構成 | **結論バンド**（結論・提言・層構造の言い切り＋結論ピル） |
| `add_matrix_2x2_slide()` | ポンチ絵 | 軸付き2×2（軸矢印・象限ラベル自動。**推奨**） |
| `add_arrow_flow_slide()` | ポンチ絵 | 横向き矢羽根フロー（コネクタ禁止の代替） |
| `add_donut_slide()` | ポンチ絵 | 構成比ドーナツ（BLOCK_ARC・×0.6 換算） |
| `add_issue_tree_slide()` | ポンチ絵 | 論点分解ロジックツリー |
| `add_process_slide()` / `add_stepup_slide()` | ポンチ絵 | 戦略提言（縦STEP／階段ロードマップ） |
| `add_cards_slide()` / `add_triangle_slide()` / `add_pyramid_slide()` / `add_compare_slide()` / `add_progress_bar_slide()` / `add_timeline_slide()` | ポンチ絵 | カード／3要素／階層／左右比較／バー／時系列 |
| `add_table_slide()` / `add_hypothesis_slide()` | データ | 表（DEEP ヘッダ＋判定色・読み取り/決め手必須）／仮説検証 |
| `add_shift_slide()` | 章構成 | 重心移動 PAST→PRESENT（エグゼクティブサマリー） |
| `add_convergence_slide()` | 章構成 | N手法が同じ結論へ収束（クロス統合の束ねスライド） |
| `add_priority_actions_slide()` / `add_action_items_slide()` | 章構成 | 優先度別アクション／☐チェックリスト（結論・提言部の末尾） |
| `add_title_slide()` / `add_section_slide()` / `add_closing_slide()` | 構造 | 表紙(DEEP)／章扉(ライト)／裏表紙(DEEP・Thank You なし。フッター呼ばない) |
| `add_toc_slide()` / `add_dual_panel_slide()` / `add_narrative_slide()` / `add_image_slide()` / `add_recommendation_slide()` / `add_matrix_slide()` | 補助 | 目次／2カラム比較／テキスト主体／全画面図／推奨アクション／旧式2×2 |
| `add_annotation_cards()` / `add_callout_box()` / `chip()` | 部品 | 注釈カードレール（図の見方/読み取り/別の見方/示唆）／注意・決め手ボックス／ラベルピル |

---

## Section 4: スライド構成ルール

### 枚数・比率ルール

| ルール | 基準 |
|--------|------|
| **最低枚数** | 25枚以上（1,000件超の分析は30枚以上推奨） |
| **チャート+注釈スライド** | 全体の50%以上 |
| **テキストのみスライド** | 全体の10%以下（エグゼクティブサマリー + 結論のみ） |
| **ポンチ絵パターン** | 画像がない全スライドにいずれかのパターンを適用 |
| **同一タイプ連続制限** | 同じスライドタイプ3枚連続禁止 |

### 内容品質ルール

| ルール | 基準 |
|--------|------|
| **タイトル** | 結論型（原則4）。数値を含む。1文で言い切る（波ダッシュ「～」禁止・補足は「—」/句点）。ラベル型禁止 |
| **リード文** | データに基づく核心主張の完結文（40〜90字）。全コンテンツスライドに必須 |
| **注釈テキスト** | 4-6項目、各1-2行（各40〜70字）、9〜10pt |
| **出所** | データを含む全スライドに `add_source_label()` |
| **フッター** | 全コンテンツスライドに `add_bottom_bar_and_footer()` |
| **フッター文言** | `APOLLO｜レポートタイトル｜日付`＋右頁番号（ブランドは "APOLLO" — "APOLLO CAPCOM" は不可） |
| **長文禁止** | 3行を超えるテキストブロックは、カード or 箇条書きに分割 |

### レイアウトルール

| ルール | 基準 |
|--------|------|
| **余白禁止** | コンテンツは下端フッター線まで埋める |
| **2層サーフェス** | 地は BG オフホワイト・コンテンツは白角丸カード（CARD＋CARD_LINE 枠）。DEEP 濃紺は表紙・裏表紙・結論ピル・表ヘッダのみ。章扉はライト（BG 地） |
| **フッター** | 全スライド（表紙/章扉/裏表紙除く）にヘアライン＋新形式フッター |
| **タイトル下線** | 全コンテンツスライドに ACCENT 短下線（0.55in × 2.25pt。全幅下線は使わない） |
| **セクション番号** | 章扉にゴースト番号（230pt・GHOST 色・Black・右寄せ） |
| **色の載せ方** | CAT はバー・チップ・ティント地のみ。文字に CAT 彩色を載せない。白文字は CHIP/DEEP 地のみ |
| **下端帯** | 全幅 ACCENT 帯（0.16in）は表紙・裏表紙のみ |

---

## Section 5: 推奨スライドシーケンス（25-38枚）

```
# --- 導入 ---
 1. タイトルスライド（表紙・DEEP基調）               add_title_slide
 2. 目次（Agenda）                                  add_toc_slide
 3. セクション: エグゼクティブサマリー               add_section_slide(1, ...)
 4. KPIダッシュボード                                add_kpi_slide
 5. 重心移動（PAST→PRESENT）                          add_shift_slide
    └ 主役交代・重心移動が結論なら add_narrative_slide の代わりに使う

# --- NEBULA 環境分析 ---
 6. セクション: NEBULA環境分析                       add_section_slide(2, ...)
 7. マクロ環境チャート + 注釈                         add_chart_text_slide
 8. 政策タイムライン                                  add_timeline_slide

# --- ATLAS 基本統計 ---
 9. セクション: ATLAS基本統計                        add_section_slide(3, ...)
10. 出願推移チャート + 注釈                           add_chart_text_slide
    └ 公開遅延の注意は add_callout_box で明示
11. 出願人ランキング（テーブル or デュアルパネル）     add_table_slide / add_dual_panel_slide

# --- CORE 分類分析 ---
12. セクション: CORE分類分析                         add_section_slide(4, ...)
13. 分類結果チャート + 注釈                           add_chart_text_slide

# --- Saturn V クラスタ分析 ---
14. セクション: Saturn Vクラスタ分析                  add_section_slide(5, ...)
15. ランドスケープ全体図                              add_chart_text_slide
    └ 右レールは「図の見方/読み取り/別の見方/示唆」の4部構成（add_annotation_cards）
16. クラスタ動態マップ                                add_chart_text_slide
17. ノイズ分析 + ピラミッド                           add_pyramid_slide
18. クラスタ詳細（カード or テーブル）                 add_cards_slide / add_table_slide
19. ミクロ分析（代表特許テーブル）                     add_table_slide

# --- MEGA 動態分析 ---
20. セクション: MEGA動態分析                         add_section_slide(6, ...)
21. MEGA 4象限マトリクス                              add_matrix_slide
22. 成長率プログレスバー                              add_progress_bar_slide

# --- Explorer/CREW ネットワーク ---
23. セクション: Explorer/CREWネットワーク            add_section_slide(7, ...)
24. キーワード共起ネットワーク                        add_chart_text_slide
25. 出願人ネットワーク（CREW）                       add_chart_text_slide

# --- NEBULA 学術分析（データがある場合）---
26. セクション: 学術分析                              add_section_slide(8, ...)
27. 学術ランドスケープ + 比較                         add_compare_slide

# --- 統合分析 ---
28. セクション: クロスモジュール分析                   add_section_slide(9, ...)
29. N手法収束（クロス統合の束ねスライド）              add_convergence_slide
30. 技術-市場-政策トライアングル                      add_triangle_slide
31. 仮説検証テーブル                                  add_hypothesis_slide
    └ 判定の分かれ目は add_callout_box（決め手）で言語化

# --- 戦略提言 ---
32. セクション: 戦略提言                              add_section_slide(10, ...)
33. 結論バンド（結論の層構造を言い切る）               add_bands_slide
34. ロードマップ（ステップアップ）                     add_stepup_slide
35. 優先度別アクション（高/中/低＋期間）               add_priority_actions_slide
    └ 優先度＋期間＋詳細だけで足りるなら add_recommendation_slide でも可

# --- 締め ---
36. Appendix（データテーブル）                        add_table_slide
37. アクションアイテム（☐チェックリスト・裏表紙直前）  add_action_items_slide
38. クロージング（裏表紙・Thank You なし）             add_closing_slide
```

### シーケンス適用ガイドライン

- 上記は最大構成。分析データが不足するセクションはスキップしてよい
- NEBULA学術分析はOpenALEXデータがある場合のみ
- CORE分類分析はルール設定済みの場合のみ
- 最低限必須: タイトル + KPI + ATLAS(1枚) + Saturn V(2枚) + MEGA(1枚) + 仮説検証 + クロージング = 約12枚
- 推奨: 25枚前後。データが豊富な場合は30-35枚

---

## Section 6: コンテンツ作成ルール

### タイトルの書き方

```
OK: 「出願件数はCAGR 20%で成長—2022年ピーク後は選択と集中フェーズへ」
OK: 「上位5クラスタが全体の58%を占有。技術集中化が加速し差別化領域の特定が急務」
NG: 「上位5クラスタが全体の58%を占有 ～技術集中化が加速」（波ダッシュ「～」は使わない）
NG: 「クラスタ分析結果」（ラベル型）
NG: 「出願動向について」（内容不明）
```

必須要素:
1. 数値を1つ以上含む
2. 結論・示唆を述べる（「〜が必要」ではなく「差別化領域の特定が急務」のように言い切る）
3. **波ダッシュ「～」は使わない**。補足が要れば全角ダッシュ「—」か句点で短い2文に分ける

### 注釈の書き方

```
OK: "2015-2022年は**CAGR 20.3%**で急成長"          <- 1行、数値あり
OK: "A社が**シェア18%**で首位維持"                   <- 1行、数値あり
NG: "2015年から2022年にかけて、出願件数は年平均
     成長率20.3%で急成長を遂げた。特に2020年以降は
     コロナ禍にもかかわらず出願が加速した。"          <- 3行超、長すぎる
```

必須要素:
1. 各注釈は1-2行（各40〜70字）
2. **太字**で数値や重要語を強調
3. 4-6項目に絞る（最後の1項目は締め文＝So What）
4. 各項目にデータポイントを含む

### リード文の書き方

```
OK: "2015-2023年の出願推移を分析。**ピーク年の2022年（263件）**以降は微減傾向"
NG: "出願推移の分析結果を示す"（具体性なし）
```
