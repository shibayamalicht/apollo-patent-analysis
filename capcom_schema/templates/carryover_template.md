<!-- これは reports/_carryover.md のテンプレート。Phase A 開始時にこのファイルを
     reports/_carryover.md にコピーして使う（既にあれば上書きしない）。 -->
# CAPCOM 分析ハンドオフ日誌（内部作業メモ・レポート転載厳禁）

<!-- 役割: フェーズ分割（別スレッド実行／コンテキスト枯渇）で「分析の記憶」を失わないための引き継ぎ台帳。
     成果物 *_deep_dive.typ（本文）と _phase_a_decisions.json（機械可読の確定値）が捨象する
     "なぜそう判断したか・精読メモ・クロス検証過程・Web出所" を保持する。
     本ファイルは執筆者(AIエージェント)の参照専用。内部識別子（モジュールJSON名・内部フィールド名・
     SQ記号・Phase名・本ファイル名）を作業上そのまま書いてよいが、report.typ /
     report_executive.typ へは1文字もコピーしないこと（本文へのコピー元は *_deep_dive.typ のみ）。
     追記規律: 各追記の冒頭に [YYYY-MM-DD / Phase X / thread N] を付ける。append-only
     （書き換えてよいのは STATUS の完了チェックとメタ行のみ）。 -->

## STATUS
- 母集団タイトル:
- 完了フェーズ: [ ]A [ ]A2 [ ]B [ ]C-saturnv [ ]C-explorer [ ]C-mega [ ]C-atlas [ ]C-core [ ]C-nebula [ ]C-crew [ ]D
- 次アクション（1行）:
- 別冊フラグ: ON | OFF
- nebula_strategy.mode: execute | web_compensation | omit   <!-- 正本は _phase_a_decisions.json。ここは参照用ミラー -->

## RESUME（新スレッド起動時に最初に実行する手順）
1. `ls reports/` で到達点判定（`_phase_a_decisions.json`→A完了 / `<module>_deep_dive.typ`→そのモジュールC完了 / `report.typ`→D統合済み）
2. 本ファイルの STATUS と直近フェーズ節・WEB出所台帳・申し送りを読む（肥大時はこの4つを優先読み）
3. `reports/_phase_a_decisions.json` を読む（母集団タイプ / sub_questions / nebula_strategy / forbidden_expressions）
4. 着手フェーズの `capcom_schema/analysis/` ガイドを読み直す
5. STOP-GATE で「_carryover.md と _phase_a_decisions.json を読了し現在地を復元した」と1行報告してから着手
   → 日誌に既にある情報（AIインサイト・Evidence）は再読しない（再読はトークン枯渇の主因）

## PHASE A 記憶
- データ全体像（抽出済み数値）: 総件数 / 期間 / DB名 / 上位出願人 / 主要IPC / HHI・Entropy・Gini / CAGR 等:
- 母集団タイプ判定の根拠と迷い:
- AIインサイト読了メモ（最低8件・各「ファイル名 + 要点1-2行」）:

## PHASE B 記憶
- クロスパターン（最低5・各3-5行）: 番号 → 仮説 → 検証データ → 結論 → 採否:
- 確定代表特許（公開番号のみ列挙。タイトル・分析は deep_dive 側で展開）:

## WEB出所台帳（footnote の原本・1発見=1行・調査ヒット直後に即追記）
<!-- web_compensation モードでは 市場規模/政策・規制/学術動向/主要企業動向 を最低1行ずつ。
     Phase D で本文に主張を書く際、該当行から #footnote[サイト名 (URL), 取得日: YYYY-MM-DD] を生成し、
     「footnote化」列を「済」にする。全行が「済」＝Web情報の本文反映漏れなし。 -->

| id | カテゴリ | 主張の要旨 | サイト名/記事名 | URL | 取得日 | footnote化 | 紐付く章 |
|----|---------|-----------|----------------|-----|--------|-----------|---------|
| W1 |  |  |  | https:// |  | 未 |  |

## 判断ログ（各 STOP-GATE のユーザー決定と「なぜそう判断したか」を1行ずつ）
-

## PHASE C 進捗（本文は *_deep_dive.typ に残るのでポインタ/未解決点のみ・二重保存禁止）
- saturnv:
- explorer:
- mega:
- atlas:
- core:
- nebula:
- crew:

## 申し送り（次スレッドへの TODO・保留・本編⇄別冊の不整合など）
-
