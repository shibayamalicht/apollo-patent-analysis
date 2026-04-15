// ==================================================================
// report.typ — APOLLO CAPCOM レポートテンプレート
// Claude Code がこのファイルをコピーし、内容を展開して使用する
// コンパイル: typst compile --root ".." reports/report.typ reports/report.pdf
// ==================================================================

#import "report_style.typ": *

// --- テンプレート適用（フォント・ページ・見出し一括設定） ---
#show: apollo-report.with(
  title: "特許分析レポート",          // ← タイトルを変更
  subtitle: "APOLLO CAPCOM Analysis", // ← サブタイトルを変更
  date: "2026年XX月",                 // ← 日付を変更
  author: "APOLLO CAPCOM Analysis",
)

// --- 目次 ---
#outline(
  title: [目次],
  indent: 1.5em,
  depth: 2,
)

// ==================================================================
// ここからレポート本文を記述
// ==================================================================

= Executive Summary

#exec-summary[
  // ここにエグゼクティブサマリーを記述
  本レポートは...
]

// --- KPIダッシュボード（ページまたぎ防止）---
#kpi-dashboard(cols: 3,
  kpi-card("総特許数", "1,234", note: "2018-2024年"),
  kpi-card("クラスタ数", "12", note: "Saturn V TELESCOPE"),
  kpi-card("主要出願人", "15社", note: "HHI = 0.18"),
)

= 分析結果

== セクション1

// 本文テキストをここに記述...

#evidence-box(1, "ATLASマクロ統計")[
  // Evidence 1 の説明・分析

  // スナップショット画像を挿入する場合:
  // #snapshot-figure("../snapshots/voyager_ev1_0.png", caption: "出願推移グラフ")
]

== セクション2

// BCG風スタイルテーブルの例
#styled-table(
  columns: (auto, 1fr, auto, auto),
  header: ([順位], [出願人], [件数], [シェア]),
  [1], [トヨタ自動車], [245], [19.8%],
  [2], [本田技研工業], [198], [16.0%],
  [3], [日産自動車], [156], [12.6%],
)

#insight-box[
  // 重要な発見をハイライト
]

#note-box[
  // 補足情報や注意事項
]

= 結論と提言

// 結論と提言を記述...
