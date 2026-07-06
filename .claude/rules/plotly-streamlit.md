---
paths:
  - "pages/**"
  - "utils.py"
  - "utils_spatial.py"
  - "utils_ai.py"
  - "Home.py"
---

# Streamlit / Plotly 描画の罠（推測で直すと逆に壊す。実描画で確認してから確定する）

これらは挙動が直感に反し、過去に実際に時間を溶かした落とし穴。描画系の修正は **推測で確定せず、Claude_Preview MCP で実描画・イベント発火を確認**してから進める。

## 対数軸（最重要・過去に ATLAS/MEGA を壊してリバート）
- **`add_vline` / `add_hline` の座標は生値を渡す**。Plotly は対数軸で座標を自動 log10 変換する（plotly 6.5.2）。自分で log10 した値を渡すと二重変換でズレる。
- **対数軸の `range` は log10 単位で解釈される**。生値で `range` を設定すると軸が壊れる。対数軸では range 拡張せず `autorange` のままにする（`utils.apply_report_theme` は対数軸で拡張をスキップ済み。線形軸のみラベル見切れ防止で拡張）。
- 権利化率マップは既定で対数X ON。CAPCOM 同梱 PNG は**スナップショットを再取得しないと旧 PNG のまま**。

## クリック / 選択イベント
- **`px.imshow`（ヒートマップ）は `on_select` の点クリックが発火しない**。透明な `go.Scatter` を重ね、`hovermode='closest'` + `hoverdistance=-1` でクリック可能にする。
- **Streamlit の選択は `plotly_selected` イベント**（`plotly_click` ではない）。配線検証は合成 MouseEvent では発火しないので `gd.emit('plotly_selected', {...})` で行う。
- 投げ縄/クリックの `customdata` は `[0]` 展開で取り出す（`handle_map_click` と同方式）。

## 再描画 / キー / パフォーマンス
- **チャートの `key=` を同操作の応答内で動的に変えると `Bad message format`（setIn エラー）**。静的キーにする。
- **大量ラベル編集（数十〜百クラスタ）＋重いマップの同一ページ再描画は WebSocket が落ちる**（`Bad message format` / `SessionInfo before initialized`）。`create_label_editor_ui` は `@st.fragment` で隔離済み（セル編集はフラグメントのみ再実行）。緩和として `.streamlit/config.toml` の `[server] maxMessageSize=500`。
- **`expander` のネストは禁止**（`StreamlitAPIException: Expanders may not be nested`）。外側を `st.container(border=True)` にして内側 expander を使う。

## マップ整合（Saturn V と揃える）
- メインランドスケープは `height=1200`・aspect 1:1。**`xaxis` に `constrain="domain"` が必須**（無いと注釈の有無で範囲が広がり俯瞰図が縮む）。表示モード既定は「クラスタ領域（凸包）」。
- **凸包ズレ＝`Scattergl`/`Scatter` 混在**: 凸包（`go.Scatter`=SVG・`fill='toself'`）を重ねるマップでプロット点を `go.Scattergl`（WebGL）にすると、WebGL と SVG の座標系の差で凸包が点群からズレる。メイン俯瞰図は両方 `go.Scatter` でズレないが、**ドリルダウン（Saturn V / EAGLE）が `Scattergl` でズレていた**（2026-06-30 修正＝プロット点を `go.Scatter` に統一）。**凸包を重ねるマップはプロット点も `go.Scatter` に揃える**（サブクラスタは点数が少なく性能/lasso とも問題なし）。

## 反映 / 検証
- **`utils.py` / `utils_ai.py`（インポート済みモジュール）の変更はサーバ再起動が必要**。ブラウザ Rerun では反映されない。`pages/*.py` は Rerun で反映。
- 実描画検証は `.claude/launch.json`（`apollo` 構成）＋ Claude_Preview MCP（preview_start/screenshot/eval/stop、ポート3000）。**ユーザーの 8501 は触らない**。
