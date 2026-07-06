# Phase D 品質検証ガイド

> このファイル名はレポート本文に書かないこと（執筆者の内部参照専用）。
> Phase D の概要 → `SKILL.md` Phase D セクション ／ 用語統一 → `analysis/terminology.md`

---

## 1. 定量チェック（機械の正本は gate スクリプト）

```bash
bash capcom_schema/scripts/phase_d_gate.sh
```

**これ1本で Check 1〜37（内容量・引用数・用語・スコープ・母集団タイプ・立場・意図・反復水増し・マップ・構造化分析・裏付け 等）を統合実行し合否判定する**（SKILL.md `## 0. 絶対遵守ゲートルール` に基づく強制ゲート。個別 grep コマンドをここに再掲しない——検査ロジックの正本は gate スクリプトである）。

- 🚫 不合格時に「実質的にOK」等の質的判断で上書きしない（ゲートルール第3項）
- 🚫 行数・件数は「量」でなく「固有性」で満たす。反復・定型文量産・自明な一般論での水増しは Check 19 で自動不合格（正本: `deep_dive_guide.md`「記述品質の絶対基準」）
- PPTX（任意出力）が存在すれば Check 16 系が自動検査する

---

## 2. 品質チェックリスト（gate が見ない/見きれない項目の人間確認）

### 2.0 Phase A 母集団設計の読解（絶対遵守・AI 自己判断での省略は不合格）

- [ ] **STOP-GATE A**（query_logic 構造化読解）: 4ステップ読解＋`AskUserQuestion` 確認を実施した（→ `query_logic_reading.md` §1）
- [ ] **STOP-GATE B**（意図↔論理の整合）: 乖離を Critical/Warning/Info に分類し改善提案付きでユーザー選択を得た。「範囲と限界に明記」選択時は該当章に記載済み（→ 同 §4）
- [ ] **STOP-GATE C**（データ逆読み＋タイプ判定）: Level 2 項目・偏り警告・タイプ A/A'/B/C/D 判定をユーザー確認し `reports/_phase_a_decisions.json` に保存した（→ 同 §5、`population_type_metrics.md` §4-2）
- [ ] **STOP-GATE D**（NEBULA 戦略）: execute / web_compensation / omit をユーザー選択で確定した（gate Check 13 がモード別検証）
- [ ] **サブクエスチョン化**: `sub_questions` を保存し、本文は問い/答え形式でなく宣言調（gate Check 12。→ `terminology.md` §5-A）
- [ ] **追加的事項章**（条件付き必須）: 乖離・偏り・想定外観察があれば「分析過程で確認された追加的事項」章を設置。無ければ `user_notes` に「追加的事項なし」と記録（→ `report_structure.md`）
- [ ] **タイプ別運用**: タイプ C は出願人 HHI を算出しない等、`population_type_metrics.md` §2 の指標制約を執筆時に適用した（gate Check 11）

### 2.1 構造・完成度

- [ ] deep_dive 4ファイル以上（Saturn V/Explorer/MEGA/ATLAS）が存在し、`phase_c_gate.sh` 合格（不足なら Phase C に戻る）
- [ ] **情報ロス**: report.typ 各章 ≥ 対応 deep_dive の行数の90%（全文コピー原則 → `report_structure.md` §2）
- [ ] `_carryover.md` は参照専用で本文に転記していない（Web出所台帳は全行 footnote 化済み）
- [ ] **走査層**: 各番号セクション冒頭に `#point-lead`（散文の代替にしない）。章末 `#chapter-summary` は本文完成後に最後へ1個（位置は gate Check 25/26。→ `deep_dive_guide.md`「読みやすさ（走査層）」）

### 2.2 ミクロ分析

- [ ] 全 deep_dive にミクロ分析A（代表特許・3点セット・計15件以上）とB（上位5社・各5行以上）
- [ ] **決定的選定**: Phase C 開始時に `select_representatives.py` を実行し、ミクロ分析A は `reports/representative_patents.json` の番号のみ引用（gate Check 35。→ `deep_dive_guide.md` ミクロ分析A）

### 2.3 分析品質

- [ ] **クロスモジュール分析**: 最低5パターン（P1-P13）、各パターン固有の仮説→検証→結論で15-20行。参照モジュール・数値根拠が他パターンと重複しない（→ `cross_module.md`）
- [ ] **4層モデル**: 思考として機能し、層ラベルは本文に不露出（gate Check 14）。第1層の断定には裏付けを添える（gate Check 36。→ `common_framework.md` §1）
- [ ] **データソースマーカー**: 全 evidence-box に具体的モジュール名・手法名（→ `common_framework.md` §5）
- [ ] 水増し・工程ナレーション・過剰修辞・国籍トートロジーなし（gate Check 19/8e/17/15。正本: `deep_dive_guide.md`「記述品質の絶対基準」・`terminology.md` §6）

### 2.4 可視化・マップ（配置ルールの正本: `common_framework.md` §4）

- [ ] 全章に `#snapshot-figure` 最低1枚。撮った全マップを掲載・分析（gate Check 20。種別不明でも接頭辞→画像内容で判定し必ず割当 → `map_reading.md` 冒頭・`deep_dive_guide.md`「マップ割当」）
- [ ] 1枚1行・全幅・見出し直後羅列禁止・隣接段落で図を名指し（gate Check 23/28/29/29b）
- [ ] ドリルダウンは `data/saturnv_drilldown_*.json` を読み、①拡大クラスタ ②サブクラスタ ③代表特許を各図に（gate Check 29c）
- [ ] ワードクラウドは上位語を『語（N回）』形式で最低5語列挙＋含意（gate Check 27。→ `map_reading.md` §6）
- [ ] 壁テキストなし: 統合・サマリ節は「統合表→散文」の順、地の文の連続は4段落まで（gate Check 37。→ `report_structure.md`・`common_framework.md` §3）

### 2.5 環境分析・AIインサイト

- [ ] `prompts/` の AI インサイト（主要モジュール各1件以上・全体8件以上）が Evidence 分析に反映されている
- [ ] `nebula_strategy` のモード別要件を満たす（execute=NEBULA 章＋各章から仮説参照／web_compensation=外部環境分析章＋4カテゴリ＋footnote／omit=範囲と限界に注記。gate Check 13）
- [ ] ノイズ分析4セクション・クラスタ動態4象限・多様性3指標（HHI/Entropy/Gini）・学術-特許クロス（execute 時）を含む（→ `deep_dive_guide.md` 新規分析要件）

### 2.6 スコープ・立場（正本: `terminology.md` §6）

- [ ] 「本母集団では」等の限定修飾が要所（エグゼクティブサマリー・範囲と限界・各章冒頭・提言）にある（gate Check 10）
- [ ] 立場（narrative_stance）に応じた呼称・提言の型。関係性立場では own_company の対比＋相手の立場から見た合理性の点検（gate Check 11s/11s'/32）

### 2.7 構造化分析（正本: `structured_techniques.md`。gate は有無と裏付けのみ・質は人間が監査）

- [ ] 主要結論に「結論の検証（別解釈＋決め手）」— 別解釈がかかし（支持のない弱い説）でないか**目視**（gate Check 30/30b）
- [ ] 主要結論1〜3個に「結論の前提と見直しのサイン」— 前提が本当に結論の要か**目視**（gate Check 31/31b）
- [ ] 仮説検証サマリー冒頭に比較検証表1枚＋直後に採用結論の明示（環1）。提言が採用結論を名指し（環2・3。gate Check 33/34）
- [ ] 読者向け用語（別解釈/決め手/見直しのサイン）。技法名の露出なし（gate Check 8f。→ `terminology.md` §2-F）

### 2.8 結論・付録・別冊

- [ ] 戦略的提言4サブセクション・推奨アクション5件以上（優先度＋時間軸＋根拠）
- [ ] 付録A（分析条件）・B（用語解説）・C（Web出所一覧・実施時）・D（検索式・指定時のみ）が `report_structure.md` §5-6 のテンプレどおり
- [ ] Web 出所は `#footnote[サイト名, 完全URL, 取得日]` の3要素（gate Check 6。→ `data_notes.md` §3）
- [ ] 仮説検証サマリーで全仮説を判定付きで回収（✅/❌/⚠️/❓＋根拠＋未検証の理由）
- [ ] 別冊（生成時）: 各要点に「別解釈と決め手」、末尾に「見直しのサイン」ボックス（gate Check 18/18b。→ `executive_summary_guide.md`）

---

## 3. 推奨項目

- [ ] マップ読解が5ステップ水準（→ `map_reading.md`）／モジュール間矛盾の考察／Mission Objective への明確な回答／提言の具体性（技術名・企業名・時間軸）／Evidence の過半を活用

## 4. 不合格時の対応フロー

1. gate の FAIL 項目を修正 → 再実行で合格確認（deep_dive 不足は Phase C へ、情報ロスは省略復元、引用不足は `representative_patents.json` から補完、図不足は `ls snapshots/` で追加）
2. WARN は「注意の誘導」——機械は正誤を裁かない。指摘箇所を目視し、直すか、妥当な理由があれば残してよい
