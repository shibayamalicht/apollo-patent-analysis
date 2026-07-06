# -*- coding: utf-8 -*-
"""APOLLO CAPCOM スライド生成ヘルパーライブラリ。

PPTX 生成スクリプトから `from apollo_slides import *` で使う。
（旧来はこの仕様を `slides_spec.md` からコピーしていたが、import 運用へ切替）

各ヘルパー（`add_*` スライド関数・`_apply_font` 等のコア関数）の
座標・色・フォントは仕様書 `slides_spec.md` のロジックをそのまま温存している。

呼び出し側で必要な前提:
    from pptx import Presentation
    prs = Presentation(TEMPLATE_OR_OWN_TEMPLATE)
    blank = prs.slide_layouts[6]
    SNAP = "<セッションフォルダ>/snapshots"   # 必要なら本モジュールの SNAP を上書き
"""

from pptx import Presentation  # noqa: F401  （呼び出しスクリプトが prs を作る際に使う）
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_AUTO_SIZE, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
from lxml import etree
import os
import re
import math

# =============================================================================
# カラーパレット（Section 1）
# =============================================================================
NAVY = RGBColor(0x1B, 0x2A, 0x4A)       # タイトル、強調テキスト、セクション背景
BLUE = RGBColor(0x2E, 0x50, 0x90)       # セクションヘッダー背景
ACCENT = RGBColor(0x3B, 0x7D, 0xD8)     # アクセントバー、強調要素
DARK_GRAY = RGBColor(0x33, 0x33, 0x33)  # 本文テキスト
MEDIUM_GRAY = RGBColor(0x66, 0x66, 0x66)  # 補足テキスト、キャプション
LIGHT_GRAY = RGBColor(0xF2, 0xF2, 0xF2)  # テーブルゼブラストライプ、枠背景
BORDER_GRAY = RGBColor(0xCC, 0xCC, 0xCC)  # テーブル罫線、区切り線
KEY_MSG_BG = RGBColor(0xE8, 0xF0, 0xFE)  # 強調ボックス背景
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
RED_ACCENT = RGBColor(0xD6, 0x45, 0x45)   # 警告、マイナス指標
GREEN_ACCENT = RGBColor(0x2E, 0x8B, 0x57)  # ポジティブ指標
AMBER = RGBColor(0xD4, 0xA0, 0x17)        # 注意指標
GHOST_NAVY = RGBColor(0x2A, 0x3A, 0x5A)   # セクションゴースト番号（NAVYより少し明るい）

# ボトムバー定数
BOTTOM_BAR_HEIGHT = 4  # px（ポイント換算: Emu(50800)）
BOTTOM_BAR_Y = 6.92    # Inches — フッター線の直上

# =============================================================================
# フォント設定（Noto Sans JP・多段ウェイト）
# =============================================================================
FONT_FAMILY = "Noto Sans JP"            # 既定（Regular）
# 役割別ウェイト。Noto Sans JP の静的ウェイト名（未インストール時は近いウェイトに自動フォールバック）。
WEIGHT_FAMILY = {
    "light":    "Noto Sans JP Light",     # キャプション・出典・フッター（控えめ）
    "regular":  "Noto Sans JP",           # 本文・注釈
    "medium":   "Noto Sans JP Medium",    # 小見出し・サブメッセージ・表ヘッダー
    "semibold": "Noto Sans JP SemiBold",  # 強調本文
    "bold":     "Noto Sans JP",           # 太字は bold フラグで（Bold 相当）
    "black":    "Noto Sans JP Black",     # 表紙・スライドタイトル・大数値・章番号
}
JA_FONT = FONT_FAMILY                    # 後方互換
LATIN_FONT = FONT_FAMILY

A_NS = 'http://schemas.openxmlformats.org/drawingml/2006/main'

# =============================================================================
# パス定数
# =============================================================================
# テンプレートPPTX（呼び出しスクリプトの位置を基準に解決する想定の相対パス）。
# 本モジュールでは import 時に Presentation を生成しない（生成は呼び出し側の責務）。
TEMPLATE = os.path.join(os.path.dirname(__file__), "../../capcom_schema/templates/apollo_template.pptx")
# スナップショット画像フォルダ。fit_image / add_chart_text_slide 等が相対パス画像を
# 解決する際の基点。CAPCOM セッション内の相対パス "snapshots" を既定にする。
# 呼び出しスクリプト側で実フォルダ（例: os.path.join(session_dir, "snapshots")）に
# 上書きしてよい。
SNAP = "snapshots"


# =============================================================================
# Section 2: コアユーティリティ — フォント・禁則ヘルパー
# =============================================================================
def _apply_font(run, weight=None):
    """runにデュアルフォント（欧文 + 日本語）+ 言語 + ウェイトを設定する。

    weight: None/"regular"/"medium"/"semibold"/"bold"/"black"/"light"。
      - None: 既定（Noto Sans JP Regular）。呼び出し側の bold フラグはそのまま尊重。
      - 名前付きウェイト（light/medium/semibold/black 等）: その専用ファミリを使い、
        bold フラグは使わない（名前付きウェイトと bold の二重指定を避ける）。
      - "bold": Regular ファミリ + bold フラグ（Bold 相当）。
    """
    fam = WEIGHT_FAMILY.get(weight, FONT_FAMILY) if weight else FONT_FAMILY
    run.font.name = fam
    if weight in ("light", "regular", "medium", "semibold", "black"):
        run.font.bold = False            # 名前付きウェイトでは bold を使わない
    elif weight == "bold":
        run.font.bold = True
    rPr = run._r.get_or_add_rPr()
    rPr.set('lang', 'ja-JP')
    rPr.set('altLang', 'en-US')
    ea = rPr.find(f'{{{A_NS}}}ea')
    if ea is None:
        ea = etree.SubElement(rPr, f'{{{A_NS}}}ea')
    ea.set('typeface', fam)


def _apply_kinsoku(paragraph):
    """段落に日本語禁則処理を設定する"""
    pPr = paragraph._p.get_or_add_pPr()
    pPr.set('eaLnBrk', '1')
    pPr.set('hangingPunct', '1')


# =============================================================================
# テキストエンジン
# =============================================================================
def add_rich_runs(paragraph, text, base_size=Pt(14), base_color=DARK_GRAY,
                  bold_color=None, force_bold=False, line_spacing=1.4, weight=None):
    """**太字**マーカー解析 + デュアルフォント + 禁則 + 行間 + ウェイト

    weight を指定すると全 run にそのウェイト（例 "black"）を適用する。
    weight=None の場合は **マーカー** 部のみ Bold、他は force_bold に従う（従来動作）。
    """
    paragraph.clear()
    bold_color = bold_color or base_color
    _apply_kinsoku(paragraph)
    paragraph.line_spacing = line_spacing

    parts = re.split(r'(\*\*.*?\*\*)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            run = paragraph.add_run()
            run.text = part[2:-2]
            run.font.size = base_size
            run.font.bold = True
            run.font.color.rgb = bold_color
        else:
            run = paragraph.add_run()
            run.text = part
            run.font.size = base_size
            run.font.bold = force_bold
            run.font.color.rgb = bold_color if force_bold else base_color
        _apply_font(run, weight)


def set_text(p, text, size, color, bold=False, line_spacing=None, weight=None):
    """シンプルテキスト設定（デュアルフォント + 禁則 + ウェイト付き）

    weight を指定すると名前付きウェイト（light/medium/black 等）を適用する
    （bold 引数より優先）。未指定なら bold フラグで太字制御（従来動作）。
    """
    p.text = ""
    run = p.add_run()
    run.text = text
    run.font.size = size
    run.font.color.rgb = color
    run.font.bold = bold
    _apply_font(run, weight)
    _apply_kinsoku(p)
    if line_spacing:
        p.line_spacing = line_spacing


# =============================================================================
# スライドタイトル（結論型 + 下線）
# =============================================================================
def add_title_shape(slide, text, x=0.5, y=0.15, w=12.3, eyebrow=None):
    """スライドタイトル（24pt Black Navy + 全幅下線）。タイトル＝主張見出し（結論）。

    結論を 1 文で言い切る（「～」副題は使わない。必要なら全角ダッシュ「—」か
    句点で短く 2 文に分ける）。数値を必ず含める。Noto Sans JP Black を使用。

    eyebrow（任意）: タイトル直上に小さな「アイブロウ」（章/モジュール名。例
        "NEBULA / 環境分析"）を添える。Noto Sans JP Medium・10pt・ミュート色・
        字間を広げて、参考デッキの編集的な見出し階層（アイブロウ→主張見出し→
        リード文→根拠→締め文・§0.9-A0）を再現する。明朝/等幅は使わずゴシックで統一。
    Returns:
        float: タイトル下端のy座標（サブメッセージの配置基準）
    """
    # アイブロウ（任意）— タイトルの上に章/モジュール名の小ラベルを置く
    if eyebrow:
        eb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.26))
        eb.text_frame.word_wrap = True
        ep = eb.text_frame.paragraphs[0]
        set_text(ep, str(eyebrow), Pt(10), MEDIUM_GRAY, weight="medium")
        for _r in ep.runs:  # 字間を少し広げてエディトリアルなアイブロウに（spc=1/100pt）
            try:
                _r._r.get_or_add_rPr().set('spc', '180')
            except Exception:
                pass
        y = y + 0.30  # タイトルをアイブロウ分だけ下げる

    text_len = len(text)
    if text_len <= 30:
        font_size = Pt(24)
        box_h = 0.65
    elif text_len <= 50:
        font_size = Pt(22)
        box_h = 0.75
    else:
        font_size = Pt(20)
        box_h = 0.90

    txBox = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(box_h))
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    p = tf.paragraphs[0]
    add_rich_runs(p, text, base_size=font_size, base_color=NAVY,
                  bold_color=NAVY, line_spacing=1.3, weight="black")

    # 全幅下線 — ACCENT色の薄いバー
    line_y = y + box_h + 0.05
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(line_y), Inches(w), Emu(12700)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()

    return line_y + 0.08  # サブメッセージの開始y座標を返す


# =============================================================================
# ■サブメッセージ
# =============================================================================
def add_sub_message(slide, message, x=0.5, y=None, w=12.3):
    """■マーカー付きサブメッセージ（ボックス囲み、タイトル直下）

    KEY_MSG_BG背景 + 左ACCENTバーのボックスで要点を強調。
    ⚠️ `message` に先頭「■」を付けないこと（本関数が自動で付与する）。
       万一付いていても二重「■ ■」にならないよう先頭の■は除去される。
    Args:
        y: 開始y座標。Noneの場合はadd_title_shapeの戻り値を使うこと。
    Returns:
        float: ボックス下端のy座標 + マージン（コンテンツ開始位置）
    """
    # 防御: 呼び出し側が先頭に「■」を付けても二重マーカーにしない
    message = re.sub(r'^[\s　]*[■▪◾]\s*', '', message)
    if y is None:
        y = 1.00
    # 高さは安定化（1-2行は一律 0.85in でブレさせない。3行以上のみ伸ばす）。
    # テキストは枠内で上下中央に置く（短文でも上に寄らない）。
    est_lines = max(1, -(-len(message) // 46))
    box_h = 0.85 if est_lines <= 2 else 0.85 + (est_lines - 2) * 0.30

    # 背景ボックス（KEY_MSG_BG）
    box = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(box_h)
    )
    box.fill.solid()
    box.fill.fore_color.rgb = KEY_MSG_BG
    box.line.fill.background()

    # 左アクセントバー
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Emu(36576), Inches(box_h)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()

    # テキスト（枠いっぱい + 上下中央寄せ）
    txBox = slide.shapes.add_textbox(
        Inches(x + 0.20), Inches(y), Inches(w - 0.38), Inches(box_h)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE          # 上下中央（ブレ防止）
    tf.margin_top = Emu(0)
    tf.margin_bottom = Emu(0)
    p = tf.paragraphs[0]
    marker = p.add_run()
    marker.text = "■ "
    marker.font.size = Pt(16)
    marker.font.color.rgb = NAVY
    _apply_font(marker, "bold")
    parts = re.split(r'(\*\*.*?\*\*)', message)
    for part in parts:
        if not part:
            continue
        run = p.add_run()
        run.font.size = Pt(16)
        if part.startswith('**') and part.endswith('**'):
            run.text = part[2:-2]
            run.font.color.rgb = NAVY
            _apply_font(run, "bold")               # 強調は Bold
        else:
            run.text = part
            run.font.color.rgb = DARK_GRAY
            _apply_font(run, "medium")             # 本文は Medium（本文 Regular との階層）
    _apply_kinsoku(p)
    p.line_spacing = 1.4

    return y + box_h + 0.10


# =============================================================================
# ボトムバー + フッター（全スライド必須）
# =============================================================================
def add_bottom_bar_and_footer(slide, page_num=None):
    """全スライド共通: ボトムアクセントバー + フッター

    ボトムバー: NAVY色、全幅、高さ4px
    フッター: 左に "APOLLO"、右にページ番号
    タイトルスライド・セクションスライド・クロージングスライドでは呼ばない。
    """
    # ボトムアクセントバー（全幅、NAVY）
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(BOTTOM_BAR_Y),
        Inches(13.33), Emu(50800)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    # フッター区切り線
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(6.95), Inches(12.3), Emu(9525)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_GRAY
    line.line.fill.background()

    # 左: "APOLLO"
    txBox = slide.shapes.add_textbox(Inches(0.5), Inches(7.05), Inches(6.0), Inches(0.25))
    set_text(txBox.text_frame.paragraphs[0], "APOLLO", Pt(8), MEDIUM_GRAY, weight="light")

    # 右: ページ番号
    if page_num is not None:
        txBox2 = slide.shapes.add_textbox(Inches(10.5), Inches(7.05), Inches(2.3), Inches(0.25))
        p2 = txBox2.text_frame.paragraphs[0]
        set_text(p2, f"| {page_num}", Pt(8), MEDIUM_GRAY)
        p2.alignment = PP_ALIGN.RIGHT


# =============================================================================
# 画像・データソース・注釈
# =============================================================================
def fit_image(slide, image_path, max_x, max_y, max_w, max_h):
    """画像をアスペクト比保持で指定領域内に中央配置"""
    if not os.path.exists(image_path):
        return None
    img = Image.open(image_path)
    img_w, img_h = img.size
    ratio = img_h / img_w
    if max_w * ratio <= max_h:
        use_w, use_h = max_w, max_w * ratio
    else:
        use_h, use_w = max_h, max_h / ratio
    left = max_x + (max_w - use_w) / 2
    top = max_y + (max_h - use_h) / 2
    pic = slide.shapes.add_picture(
        image_path, Inches(left), Inches(top),
        width=Inches(use_w), height=Inches(use_h)
    )
    img.close()
    return pic


def add_source_label(slide, source_text, x=0.5, y=6.55, w=12.3):
    """（出所）ラベル"""
    txBox = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.35))
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    p = tf.paragraphs[0]
    set_text(p, f"（出所）{source_text}", Pt(9), MEDIUM_GRAY, weight="light")


def add_annotation_block(slide, bullets, x, y, w, h, font_size=14,
                         has_border=False, bg_color=None):
    """テキスト注釈ブロック（チャート横の分析テキスト）

    ■マーカー付き箇条書きでチャートを補足する。
    各bullet: 最大2行、14pt。全体で3-5項目を推奨。
    """
    if bg_color or has_border:
        box = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
        )
        if bg_color:
            box.fill.solid()
            box.fill.fore_color.rgb = bg_color
        else:
            box.fill.background()
        if has_border:
            box.line.color.rgb = BORDER_GRAY
            box.line.width = Emu(9525)
        else:
            box.line.fill.background()

    txBox = slide.shapes.add_textbox(
        Inches(x + 0.12), Inches(y + 0.08),
        Inches(w - 0.24), Inches(h - 0.16)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE

    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(6)
        marker = p.add_run()
        marker.text = "■ "
        marker.font.size = Pt(font_size)
        marker.font.color.rgb = NAVY
        _apply_font(marker)
        parts = re.split(r'(\*\*.*?\*\*)', item)
        for part in parts:
            if not part:
                continue
            run = p.add_run()
            if part.startswith('**') and part.endswith('**'):
                run.text = part[2:-2]
                run.font.bold = True
                run.font.color.rgb = NAVY
            else:
                run.text = part
                run.font.color.rgb = DARK_GRAY
            run.font.size = Pt(font_size)
            _apply_font(run)
        _apply_kinsoku(p)
        p.line_spacing = 1.5


def add_chart_label(slide, text, x, y, w=3.0, size=14, color=NAVY):
    """チャート小見出し（グラフ上の分類ラベル）"""
    txBox = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(0.35))
    set_text(txBox.text_frame.paragraphs[0], text, Pt(size), color, bold=True)


def _add_line(slide, x0, y0, x1, y1, color, weight_pt=1.5):
    """2点間を細い回転矩形で結ぶ（コネクタ p:cxnSp はファイル破損を起こすため使わない）。

    座標は Inches 値。線色・太さは元のコネクタの見た目を踏襲する。
    """
    dx, dy = (x1 - x0), (y1 - y0)
    length = math.hypot(dx, dy)
    angle = math.degrees(math.atan2(dy, dx))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    thick = weight_pt / 72.0  # pt → inch
    ln = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(cx - length / 2), Inches(cy - thick / 2),
        Inches(length), Inches(thick)
    )
    ln.rotation = angle
    ln.fill.solid()
    ln.fill.fore_color.rgb = color
    ln.line.fill.background()
    ln.shadow.inherit = False
    return ln


def add_chart_callout(slide, text, x, y, w=2.5,
                      arrow_to_x=None, arrow_to_y=None,
                      bg_color=None, font_size=12, border_color=None):
    """チャート上の吹き出し注釈（画像の上にオーバーレイ配置）"""
    bg_color = bg_color or WHITE
    border_color = border_color or NAVY

    chars_per_line = int(w * 7)
    num_lines = max(1, -(-len(text) // chars_per_line))
    h = 0.15 + num_lines * 0.28

    box = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    box.fill.solid()
    box.fill.fore_color.rgb = bg_color
    box.line.color.rgb = border_color
    box.line.width = Emu(12700)

    txBox = slide.shapes.add_textbox(
        Inches(x + 0.08), Inches(y + 0.04), Inches(w - 0.16), Inches(h - 0.08)
    )
    tf = txBox.text_frame
    tf.word_wrap = True
    tf.auto_size = MSO_AUTO_SIZE.NONE
    add_rich_runs(tf.paragraphs[0], text, base_size=Pt(font_size),
                  base_color=NAVY, bold_color=NAVY, line_spacing=1.3)

    if arrow_to_x is not None and arrow_to_y is not None:
        cx = x + w / 2
        cy = y + h / 2
        # コネクタ（p:cxnSp）はファイル破損を起こすため、回転矩形で引き出し線を描く
        _add_line(slide, cx, cy, arrow_to_x, arrow_to_y, border_color, weight_pt=1.0)

    return box


def add_highlight_circle(slide, x, y, w=0.5, h=0.5, color=None):
    """チャート上のハイライト丸囲み"""
    color = color or RED_ACCENT
    oval = slide.shapes.add_shape(
        MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    oval.fill.background()
    oval.line.color.rgb = color
    oval.line.width = Emu(19050)
    return oval


# =============================================================================
# Section 3: スライドタイプ（15種）
# =============================================================================

# 3.1 タイトルスライド（表紙）
def add_title_slide(prs, title, subtitle, date, blank):
    """表紙 — Navy背景 + アクセントライン + APOLLOロゴ"""
    slide = prs.slides.add_slide(blank)
    # Navy背景（スライド全面）
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY

    # アクセントライン（左上）
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1.2), Inches(1.8), Inches(2.0), Emu(27432)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = ACCENT
    line.line.fill.background()

    # "APOLLO" ロゴテキスト（左上に小さく）
    logo = slide.shapes.add_textbox(Inches(1.2), Inches(1.2), Inches(3), Inches(0.5))
    set_text(logo.text_frame.paragraphs[0], "APOLLO", Pt(14), ACCENT, bold=True)

    # タイトル（36pt White Bold）
    txBox = slide.shapes.add_textbox(Inches(1.2), Inches(2.1), Inches(11), Inches(2))
    tf = txBox.text_frame
    tf.word_wrap = True
    set_text(tf.paragraphs[0], title, Pt(36), WHITE, bold=True, line_spacing=1.2)

    # サブタイトル
    txBox2 = slide.shapes.add_textbox(Inches(1.2), Inches(4.2), Inches(11), Inches(1))
    set_text(txBox2.text_frame.paragraphs[0], subtitle, Pt(18), RGBColor(0xAA, 0xBB, 0xDD))

    # 日付
    txBox3 = slide.shapes.add_textbox(Inches(1.2), Inches(5.5), Inches(11), Inches(0.5))
    set_text(txBox3.text_frame.paragraphs[0], date, Pt(13), RGBColor(0x88, 0x99, 0xBB))

    # ボトムライン（ACCENT、全幅）
    bot = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.1), Inches(13.33), Emu(27432)
    )
    bot.fill.solid()
    bot.fill.fore_color.rgb = ACCENT
    bot.line.fill.background()
    return slide


# 3.2 セクション区切り（ゴースト番号付き）
def add_section_slide(prs, section_num, title, blank, subtitle=None):
    """セクション区切り — Navy背景 + ゴースト番号(180pt, 半透明)"""
    slide = prs.slides.add_slide(blank)
    # Navy背景（全面）
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY

    # ゴースト番号（180pt、半透明 — NAVYより少し明るい色で透過効果）
    ghost = slide.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(5), Inches(3.5))
    tf_g = ghost.text_frame
    p_g = tf_g.paragraphs[0]
    run_g = p_g.add_run()
    run_g.text = f"{section_num:02d}"
    run_g.font.size = Pt(180)
    run_g.font.color.rgb = GHOST_NAVY
    run_g.font.bold = True
    _apply_font(run_g)

    # 左アクセントバー
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(1.0), Inches(3.0), Emu(36576), Inches(2.0)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = ACCENT
    bar.line.fill.background()

    # セクションタイトル（32pt White Bold）
    txBox = slide.shapes.add_textbox(Inches(1.3), Inches(3.2), Inches(11), Inches(1.5))
    tf = txBox.text_frame
    tf.word_wrap = True
    set_text(tf.paragraphs[0], title, Pt(32), WHITE, bold=True, line_spacing=1.2)

    # サブタイトル（省略可）
    if subtitle:
        txBox2 = slide.shapes.add_textbox(Inches(1.3), Inches(4.8), Inches(11), Inches(0.8))
        set_text(txBox2.text_frame.paragraphs[0], subtitle, Pt(16), RGBColor(0xCC, 0xDD, 0xEE))

    # ボトムライン
    bot = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.1), Inches(13.33), Emu(27432)
    )
    bot.fill.solid()
    bot.fill.fore_color.rgb = ACCENT
    bot.line.fill.background()
    return slide


# 3.3 チャート+テキスト注釈スライド（主力 — 50%以上）
def add_chart_text_slide(prs, title, sub_message, image_path, annotations, blank,
                         caption=None, chart_label=None, text_side="right",
                         chart_ratio=0.60, source=None, page_num=None, eyebrow=None):
    """チャート主体 + テキスト注釈 — 主力スライドタイプ（§0.9-A0 の主張骨格を載せる）

    Args:
        title: 主張見出し（結論性のある名詞句）
        sub_message: リード文（核心主張の完結した一文・数値込み）
        annotations: 根拠の完結文リスト（5語の断片でなく各1-2行・最大5項目）。
                     **最後の1項目は締め文（So What の一文）**にする
        eyebrow: タイトル直上のアイブロウ（章/モジュール名。例 "NEBULA / 環境分析"）
        text_side: "right" or "left"
        chart_ratio: チャート側の幅比率（0.55-0.65）
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title, eyebrow=eyebrow)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    content_w = 12.3
    content_x = 0.5
    gap = 0.3
    chart_w = content_w * chart_ratio - gap / 2
    text_w = content_w * (1 - chart_ratio) - gap / 2
    remaining_h = 6.5 - content_y  # ボトムバーまで使い切る

    if text_side == "right":
        chart_x = content_x
        text_x = content_x + chart_w + gap
    else:
        text_x = content_x
        chart_x = content_x + text_w + gap

    # チャート小見出し
    if chart_label:
        add_chart_label(slide, chart_label, chart_x, content_y, chart_w)
        img_y = content_y + 0.35
        img_h = remaining_h - 0.65
    else:
        img_y = content_y
        img_h = remaining_h - 0.3

    # チャート画像（領域を埋める）
    full_path = os.path.join(SNAP, image_path) if not os.path.isabs(image_path) else image_path
    fit_image(slide, full_path, max_x=chart_x, max_y=img_y, max_w=chart_w, max_h=img_h)

    # キャプション
    if caption:
        txBox = slide.shapes.add_textbox(Inches(chart_x), Inches(content_y + remaining_h - 0.25),
                                          Inches(chart_w), Inches(0.25))
        set_text(txBox.text_frame.paragraphs[0], caption, Pt(10), MEDIUM_GRAY)
        txBox.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # テキスト注釈（14pt、3-5項目）
    add_annotation_block(slide, annotations[:5], text_x, content_y,
                         text_w, remaining_h - 0.2)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.4 KPIダッシュボード
def add_kpi_slide(prs, title, sub_message, kpis, blank,
                  source=None, page_num=None):
    """KPIダッシュボード — 動的カードレイアウト

    kpis: [{"label":"総特許件数", "value":"1,176", "unit":"件", "trend":"↑12%"}, ...]
    4個以下 = 1行配置、5-8個 = 2行配置
    各カード: アクセント左バー + ラベル(小Gray) + 値(大Navy Bold) + 単位(小Gray)
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    n = len(kpis)
    available_w = 11.5
    start_x = 0.9
    gap = 0.2

    if n <= 4:
        # 1行配置
        n_cols = n
        n_rows = 1
    else:
        # 2行配置
        n_cols = min(4, (n + 1) // 2)
        n_rows = 2

    card_w = (available_w - gap * (n_cols - 1)) / n_cols
    available_h = 6.5 - content_y
    row_gap = 0.2
    card_h = (available_h - row_gap * (n_rows - 1)) / n_rows
    card_h = min(card_h, 2.8)  # 上限

    for idx, kpi in enumerate(kpis):
        row = idx // n_cols
        col = idx % n_cols
        x = start_x + col * (card_w + gap)
        y = content_y + row * (card_h + row_gap)

        # カード背景
        card = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
            Inches(card_w), Inches(card_h)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_GRAY
        card.line.color.rgb = BORDER_GRAY
        card.line.width = Emu(9525)

        # 左アクセントバー
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
            Emu(36576), Inches(card_h)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = ACCENT
        bar.line.fill.background()

        # ラベル（小、Gray・Medium）
        txL = slide.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.12),
                                        Inches(card_w - 0.3), Inches(0.25))
        set_text(txL.text_frame.paragraphs[0], kpi["label"], Pt(10), MEDIUM_GRAY, weight="medium")

        # 値（大数値コールアウト、Navy・Black）
        txV = slide.shapes.add_textbox(Inches(x + 0.15), Inches(y + 0.4),
                                        Inches(card_w - 0.3), Inches(0.7))
        p = txV.text_frame.paragraphs[0]
        run = p.add_run()
        run.text = kpi["value"]
        run.font.size = Pt(32)
        run.font.color.rgb = NAVY
        _apply_font(run, "black")

        # トレンド矢印
        if kpi.get("trend"):
            trend = kpi["trend"]
            if "+" in trend or "UP" in trend.upper():
                tc = GREEN_ACCENT
            elif "-" in trend or "DOWN" in trend.upper():
                tc = RED_ACCENT
            else:
                tc = MEDIUM_GRAY
            run2 = p.add_run()
            run2.text = f" {trend}"
            run2.font.size = Pt(14)
            run2.font.color.rgb = tc
            _apply_font(run2)

        # 単位（小、Gray）
        txU = slide.shapes.add_textbox(Inches(x + 0.15), Inches(y + card_h - 0.4),
                                        Inches(card_w - 0.3), Inches(0.25))
        set_text(txU.text_frame.paragraphs[0], kpi.get("unit", ""), Pt(9), MEDIUM_GRAY)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.5 カードスライド（3-4枚並列）
def add_cards_slide(prs, title, sub_message, cards, blank,
                    source=None, page_num=None):
    """カード並列表示 — 3-4枚のカードを横並び

    cards: [{"header":"クラスタA", "body":"説明テキスト", "color":NAVY}, ...]
    ヘッダー: 色付き背景 + 白テキスト
    ボディ: LIGHT_GRAY背景 + DARK_GRAYテキスト
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    n = len(cards)
    gap = 0.25
    total_w = 12.3
    card_w = (total_w - gap * (n - 1)) / n
    card_h = 6.5 - content_y  # 下端まで使い切る
    header_h = 0.45
    colors = [NAVY, BLUE, ACCENT, GREEN_ACCENT, RED_ACCENT, AMBER]

    for i, card in enumerate(cards):
        x = 0.5 + i * (card_w + gap)
        color = card.get("color", colors[i % len(colors)])

        # ヘッダー（色付き背景 + 白テキスト）
        hdr = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(content_y),
            Inches(card_w), Inches(header_h)
        )
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = color
        hdr.line.fill.background()

        txH = slide.shapes.add_textbox(Inches(x + 0.1), Inches(content_y + 0.05),
                                        Inches(card_w - 0.2), Inches(header_h - 0.1))
        set_text(txH.text_frame.paragraphs[0], card["header"], Pt(14), WHITE, bold=True)
        txH.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        # ボディ（LIGHT_GRAY背景）
        body_y = content_y + header_h
        body_h = card_h - header_h
        bdy = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(body_y),
            Inches(card_w), Inches(body_h)
        )
        bdy.fill.solid()
        bdy.fill.fore_color.rgb = LIGHT_GRAY
        bdy.line.color.rgb = BORDER_GRAY
        bdy.line.width = Emu(9525)

        txB = slide.shapes.add_textbox(Inches(x + 0.12), Inches(body_y + 0.1),
                                        Inches(card_w - 0.24), Inches(body_h - 0.2))
        tf = txB.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE   # 箱内テキストを上下中央（スカスカ防止・§0.9-G）

        body_text = card.get("body", "")
        if isinstance(body_text, list):
            for j, item in enumerate(body_text):
                p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                add_rich_runs(p, f"・{item}", base_size=Pt(12), base_color=DARK_GRAY,
                              bold_color=NAVY, line_spacing=1.4)
        else:
            add_rich_runs(tf.paragraphs[0], body_text, base_size=Pt(13),
                          base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.4)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.5b 2×2 マトリクススライド（軸付き・コンサル定番）
def add_matrix_2x2_slide(prs, title, sub_message, x_axis, y_axis, quadrants, blank,
                         source=None, page_num=None):
    """2×2 マトリクス（軸付き）。

    x_axis: {"label": "出願数（活動量）", "low": "少", "high": "多"}
    y_axis: {"label": "成長率（CAGR）", "low": "低", "high": "高"}
    quadrants: 4要素のリスト [左上, 右上, 左下, 右下]
        各要素 {"label": "少数精鋭", "desc": "短い説明", "color": NAVY}（color 省略可）
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    # 描画領域（左に Y 軸ラベル、下に X 軸ラベルの余白を確保）
    left, top, right, bottom = 1.7, content_y + 0.15, 12.6, 6.05
    aw, ah = right - left, bottom - top
    gap = 0.18
    cw, ch = (aw - gap) / 2, (ah - gap) / 2
    pos = [(left, top), (left + cw + gap, top),
           (left, top + ch + gap), (left + cw + gap, top + ch + gap)]
    default_colors = [BLUE, NAVY, MEDIUM_GRAY, ACCENT]

    for i, q in enumerate(quadrants[:4]):
        qx, qy = pos[i]
        color = q.get("color", default_colors[i])
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                     Inches(qx), Inches(qy), Inches(cw), Inches(ch))
        box.fill.solid()
        box.fill.fore_color.rgb = color
        box.line.color.rgb = WHITE
        box.line.width = Pt(2)
        tb = slide.shapes.add_textbox(Inches(qx + 0.15), Inches(qy + 0.13),
                                      Inches(cw - 0.3), Inches(ch - 0.26))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE   # 箱内テキストを上下中央（スカスカ防止・§0.9-G）
        set_text(tf.paragraphs[0], q.get("label", ""), Pt(15), WHITE, bold=True)
        if q.get("desc"):
            p = tf.add_paragraph()
            add_rich_runs(p, q["desc"], base_size=Pt(11), base_color=WHITE,
                          bold_color=WHITE, line_spacing=1.3)

    # X 軸（下端・右向き矢印 + ラベル + 低/高） — コネクタではなく矢印オートシェイプ
    xa = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(left), Inches(bottom + 0.10),
                                Inches(aw), Inches(0.20))
    xa.fill.solid()
    xa.fill.fore_color.rgb = DARK_GRAY
    xa.line.fill.background()
    xl = slide.shapes.add_textbox(Inches(left), Inches(bottom + 0.33), Inches(aw), Inches(0.3))
    set_text(xl.text_frame.paragraphs[0],
             f"{x_axis.get('low','低')}　←　{x_axis.get('label','')}　→　{x_axis.get('high','高')}",
             Pt(12), DARK_GRAY, bold=True)
    xl.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # Y 軸（左端・上向き矢印 + 回転ラベル）
    ya = slide.shapes.add_shape(MSO_SHAPE.UP_ARROW, Inches(left - 0.40), Inches(top),
                                Inches(0.20), Inches(ah))
    ya.fill.solid()
    ya.fill.fore_color.rgb = DARK_GRAY
    ya.line.fill.background()
    yl = slide.shapes.add_textbox(Inches(left - 1.65), Inches(top + ah / 2 - 0.18),
                                  Inches(2.0), Inches(0.36))
    set_text(yl.text_frame.paragraphs[0],
             f"{y_axis.get('low','低')} ← {y_axis.get('label','')} → {y_axis.get('high','高')}",
             Pt(12), DARK_GRAY, bold=True)
    yl.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    yl.rotation = -90

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.5c 横向き矢羽根フロー（プロセス/因果フロー）
def add_arrow_flow_slide(prs, title, sub_message, steps, blank,
                         source=None, page_num=None):
    """横向きプロセス/因果フロー。先頭=PENTAGON、以降=CHEVRON。

    steps: [{"title":"探索", "desc":"母集団1,176件を取得"}, ...]（3〜6個）
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    n = len(steps)
    left, right = 0.5, 12.83
    W = right - left
    overlap = 0.16                      # 矢羽根の先端を次のノッチへ噛み合わせる
    cw = (W + overlap * (n - 1)) / n
    band_h = 1.35
    band_y = content_y + 0.40
    colors = [NAVY, BLUE, ACCENT, GREEN_ACCENT, AMBER]
    title_size = Pt(16) if n <= 4 else Pt(13)
    desc_top = band_y + band_h + 0.25
    desc_h = max(0.6, 6.35 - desc_top)

    for i, step in enumerate(steps):
        cx = left + i * (cw - overlap)
        color = colors[i % len(colors)]
        shape_type = MSO_SHAPE.PENTAGON if i == 0 else MSO_SHAPE.CHEVRON
        ch = slide.shapes.add_shape(shape_type, Inches(cx), Inches(band_y),
                                    Inches(cw), Inches(band_h))
        ch.fill.solid()
        ch.fill.fore_color.rgb = color
        ch.line.fill.background()

        # ラベル（白・太字・中央。先端を避けて少し左に寄せた領域）
        tb = slide.shapes.add_textbox(Inches(cx + 0.10), Inches(band_y + 0.15),
                                      Inches(cw - 0.70), Inches(band_h - 0.30))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE   # 箱内テキストを上下中央（スカスカ防止・§0.9-G）
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        set_text(p, step.get("title", ""), title_size, WHITE, bold=True)

        # 直下の説明
        if step.get("desc"):
            db = slide.shapes.add_textbox(Inches(cx + 0.12), Inches(desc_top),
                                          Inches(cw - 0.40), Inches(desc_h))
            dtf = db.text_frame
            dtf.word_wrap = True
            dtf.auto_size = MSO_AUTO_SIZE.NONE
            add_rich_runs(dtf.paragraphs[0], step["desc"], base_size=Pt(12),
                          base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.3)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.5d ドーナツチャート（構成比・BLOCK_ARC）
def add_donut_slide(prs, title, sub_message, segments, blank,
                    center_label=None, source=None, page_num=None):
    """ドーナツ図。BLOCK_ARC を構成比に応じた角度で並べる。

    segments: [{"label":"正極材料", "value":330, "color":NAVY}, ...]（3〜4推奨）
    center_label: ドーナツ中央に置く大きな数値/語（任意）。
    全弧を共通の正方形バウンディングボックスに重ね、rotation=270 で頂点(12時)始まり
    に揃える。終了角>開始角で時計回りにその扇形を塗る（OOXML: swAng=adj2-adj1）。
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    total = sum(s.get("value", 0) for s in segments) or 1
    area_top = content_y + 0.20
    area_bottom = 6.30
    R = min(1.95, (area_bottom - area_top) / 2)
    cy = (area_top + area_bottom) / 2
    cx = 3.30
    inner = 0.55
    colors = [NAVY, ACCENT, GREEN_ACCENT, AMBER, BLUE, RED_ACCENT]

    DEG = 60000.0 / 100000.0             # 実角度°→adjustment値（python-pptxの格納仕様）
    ang = 0.0                            # 実角度（度）で累積
    for i, seg in enumerate(segments):
        frac = seg.get("value", 0) / total
        ang2 = ang + frac * 360.0
        color = seg.get("color", colors[i % len(colors)])
        arc = slide.shapes.add_shape(MSO_SHAPE.BLOCK_ARC,
                                     Inches(cx - R), Inches(cy - R),
                                     Inches(2 * R), Inches(2 * R))
        arc.adjustments[0] = ang * DEG
        arc.adjustments[1] = (ang2 if ang2 < 359.999 else 359.999) * DEG
        arc.adjustments[2] = inner
        arc.rotation = 270               # 12時始まり（共通中心なので群として回転）
        arc.fill.solid()
        arc.fill.fore_color.rgb = color
        arc.line.color.rgb = WHITE
        arc.line.width = Pt(1.5)
        ang = ang2

    if center_label:
        cl = slide.shapes.add_textbox(Inches(cx - 1.1), Inches(cy - 0.45),
                                      Inches(2.2), Inches(0.9))
        ctf = cl.text_frame
        ctf.word_wrap = True
        ctf.auto_size = MSO_AUTO_SIZE.NONE
        pc = ctf.paragraphs[0]
        pc.alignment = PP_ALIGN.CENTER
        set_text(pc, center_label, Pt(22), NAVY, bold=True)

    # 凡例（右側・色見本 + ラベル + 構成比%）
    lx = cx + R + 0.9
    lw = 12.83 - lx
    n = len(segments)
    row_h = min(0.62, (area_bottom - area_top) / max(n, 1))
    ly0 = cy - (n * row_h) / 2
    for i, seg in enumerate(segments):
        color = seg.get("color", colors[i % len(colors)])
        ry = ly0 + i * row_h
        sw = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(lx), Inches(ry + 0.04),
                                    Inches(0.30), Inches(0.30))
        sw.fill.solid()
        sw.fill.fore_color.rgb = color
        sw.line.fill.background()
        pct = round(seg.get("value", 0) / total * 100)
        tb = slide.shapes.add_textbox(Inches(lx + 0.45), Inches(ry), Inches(lw - 0.45), Inches(row_h))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE   # 箱内テキストを上下中央（スカスカ防止・§0.9-G）
        add_rich_runs(tf.paragraphs[0],
                      f"**{seg.get('label','')}** — {pct}%（{seg.get('value',0)}）",
                      base_size=Pt(13), base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.2)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.5e Issue Tree / ロジックツリー（論証骨格）
def add_issue_tree_slide(prs, title, sub_message, root, branches, blank,
                         source=None, page_num=None):
    """左に論点（根）、右に分解した枝を縦に並べる2階層ロジックツリー。

    root: {"title":"なぜ権利化率に差?", "desc":"上位出願人で2倍超の開き"}
    branches: [{"title":"出願戦略の差", "desc":"量産型 vs 厳選型"}, ...]（2〜5）
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    area_top = content_y + 0.20
    area_bottom = 6.30
    mid_y = (area_top + area_bottom) / 2

    left = 0.5
    root_w = 2.9
    root_h = min(1.7, area_bottom - area_top)
    root_y = mid_y - root_h / 2
    spine_x = left + root_w + 0.95
    arrow_w = 0.55
    child_x = spine_x + arrow_w + 0.05
    child_w = 12.83 - child_x
    n = len(branches)
    gap = 0.22
    child_h = min(1.45, (area_bottom - area_top - gap * (n - 1)) / n)
    total_h = n * child_h + (n - 1) * gap
    start_y = mid_y - total_h / 2
    child_colors = [NAVY, BLUE, ACCENT, GREEN_ACCENT, AMBER]
    LINE_T = 0.035                       # 枝線の太さ（細い矩形）

    # 根（論点）ボックス
    rbox = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(left),
                                  Inches(root_y), Inches(root_w), Inches(root_h))
    rbox.fill.solid()
    rbox.fill.fore_color.rgb = NAVY
    rbox.line.fill.background()
    rtb = slide.shapes.add_textbox(Inches(left + 0.15), Inches(root_y + 0.13),
                                   Inches(root_w - 0.30), Inches(root_h - 0.26))
    rtf = rtb.text_frame
    rtf.word_wrap = True
    rtf.auto_size = MSO_AUTO_SIZE.NONE
    set_text(rtf.paragraphs[0], root.get("title", ""), Pt(15), WHITE, bold=True)
    if root.get("desc"):
        add_rich_runs(rtf.add_paragraph(), root["desc"], base_size=Pt(11),
                      base_color=WHITE, bold_color=WHITE, line_spacing=1.25)

    # 根→幹の水平枝（細い矩形）
    htrunk = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left + root_w),
                                    Inches(mid_y - LINE_T / 2),
                                    Inches(spine_x - (left + root_w)), Inches(LINE_T))
    htrunk.fill.solid()
    htrunk.fill.fore_color.rgb = MEDIUM_GRAY
    htrunk.line.fill.background()

    child_centers = [start_y + i * (child_h + gap) + child_h / 2 for i in range(n)]
    # 縦の幹（最初〜最後の枝中心を結ぶ細い矩形）
    if n > 1:
        sp_top = min(child_centers[0], mid_y)
        sp_bot = max(child_centers[-1], mid_y)
        vspine = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(spine_x - LINE_T / 2),
                                        Inches(sp_top), Inches(LINE_T), Inches(sp_bot - sp_top))
        vspine.fill.solid()
        vspine.fill.fore_color.rgb = MEDIUM_GRAY
        vspine.line.fill.background()

    for i, br in enumerate(branches):
        cyc = child_centers[i]
        cy_box = cyc - child_h / 2
        color = child_colors[i % len(child_colors)]
        # 幹→枝の右向き矢印（方向表現・オートシェイプ）
        ar = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(spine_x),
                                    Inches(cyc - 0.11), Inches(arrow_w), Inches(0.22))
        ar.fill.solid()
        ar.fill.fore_color.rgb = color
        ar.line.fill.background()
        # 枝ボックス（左に色帯 + 本文）
        cbox = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(child_x), Inches(cy_box),
                                      Inches(child_w), Inches(child_h))
        cbox.fill.solid()
        cbox.fill.fore_color.rgb = LIGHT_GRAY
        cbox.line.color.rgb = BORDER_GRAY
        cbox.line.width = Emu(9525)
        cbar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(child_x), Inches(cy_box),
                                      Inches(0.12), Inches(child_h))
        cbar.fill.solid()
        cbar.fill.fore_color.rgb = color
        cbar.line.fill.background()
        ctb = slide.shapes.add_textbox(Inches(child_x + 0.28), Inches(cy_box + 0.10),
                                       Inches(child_w - 0.45), Inches(child_h - 0.20))
        ctf = ctb.text_frame
        ctf.word_wrap = True
        ctf.auto_size = MSO_AUTO_SIZE.NONE
        set_text(ctf.paragraphs[0], br.get("title", ""), Pt(14), NAVY, bold=True)
        if br.get("desc"):
            add_rich_runs(ctf.add_paragraph(), br["desc"], base_size=Pt(12),
                          base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.25)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.6 プロセスフロー（縦STEP型）
def add_process_slide(prs, title, sub_message, steps, blank,
                      source=None, page_num=None):
    """縦STEPプロセスフロー

    steps: [{"title":"データ収集", "desc":"特許DBから1,176件を取得"}, ...]
    2個以下 = 大ボックス、3個 = 中、4個以上 = コンパクト
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    n = len(steps)
    available_h = 6.5 - content_y
    gap = 0.15
    step_h = (available_h - gap * (n - 1) - 0.5) / n  # 矢印分0.5確保
    step_h = min(step_h, 1.5)

    # フォントサイズ調整
    if n <= 2:
        title_size, desc_size = Pt(16), Pt(14)
    elif n <= 3:
        title_size, desc_size = Pt(14), Pt(13)
    else:
        title_size, desc_size = Pt(13), Pt(12)

    header_w = 2.2
    body_w = 9.8
    colors = [NAVY, BLUE, ACCENT, GREEN_ACCENT, AMBER]

    for i, step in enumerate(steps):
        sy = content_y + i * (step_h + gap + 0.15)
        color = colors[i % len(colors)]

        # 左ヘッダー（色付き）
        hdr = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(sy),
            Inches(header_w), Inches(step_h)
        )
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = color
        hdr.line.fill.background()

        txH = slide.shapes.add_textbox(Inches(0.6), Inches(sy + 0.1),
                                        Inches(header_w - 0.2), Inches(step_h - 0.2))
        tf_h = txH.text_frame
        tf_h.word_wrap = True
        p_h = tf_h.paragraphs[0]
        p_h.alignment = PP_ALIGN.CENTER
        set_text(p_h, f"STEP {i+1}", Pt(10), WHITE)
        p_t = tf_h.add_paragraph()
        p_t.alignment = PP_ALIGN.CENTER
        set_text(p_t, step["title"], title_size, WHITE, bold=True)

        # 右ボディ（LIGHT_GRAY背景）
        bdy = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.5 + header_w + 0.1), Inches(sy),
            Inches(body_w), Inches(step_h)
        )
        bdy.fill.solid()
        bdy.fill.fore_color.rgb = LIGHT_GRAY
        bdy.line.color.rgb = BORDER_GRAY
        bdy.line.width = Emu(9525)

        txB = slide.shapes.add_textbox(Inches(0.5 + header_w + 0.25), Inches(sy + 0.1),
                                        Inches(body_w - 0.3), Inches(step_h - 0.2))
        tf_b = txB.text_frame
        tf_b.word_wrap = True
        tf_b.auto_size = MSO_AUTO_SIZE.NONE
        add_rich_runs(tf_b.paragraphs[0], step.get("desc", ""),
                      base_size=desc_size, base_color=DARK_GRAY, bold_color=NAVY)

        # 下矢印（最後のステップ以外）
        if i < n - 1:
            arrow_y = sy + step_h + 0.02
            arrow = slide.shapes.add_shape(
                MSO_SHAPE.DOWN_ARROW, Inches(1.2), Inches(arrow_y),
                Inches(0.5), Inches(gap + 0.05)
            )
            arrow.fill.solid()
            arrow.fill.fore_color.rgb = color
            arrow.line.fill.background()

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.7 ステップアップスライド（階段型ロードマップ）
def add_stepup_slide(prs, title, sub_message, phases, blank,
                     source=None, page_num=None):
    """ステップアップ（階段型ロードマップ）

    phases: [{"header":"短期", "body":"基盤構築", "color":ACCENT}, ...]
    左から右へ棒の高さが上がる。3-4段を推奨。
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    n = len(phases)
    gap = 0.2
    total_w = 12.3
    bar_w = (total_w - gap * (n - 1)) / n
    base_y = 6.5  # ボトムバー直上
    max_h = base_y - content_y - 0.2
    colors = [ACCENT, BLUE, NAVY, GREEN_ACCENT]

    for i, phase in enumerate(phases):
        x = 0.5 + i * (bar_w + gap)
        # 高さを段階的に上げる（最小50%、最大100%）
        ratio = 0.5 + 0.5 * (i / max(n - 1, 1))
        bar_h = max_h * ratio
        y = base_y - bar_h
        color = phase.get("color", colors[i % len(colors)])

        # ヘッダー部（上部、色付き）
        header_h = min(0.5, bar_h * 0.25)
        hdr = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(y),
            Inches(bar_w), Inches(header_h)
        )
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = color
        hdr.line.fill.background()

        txH = slide.shapes.add_textbox(Inches(x + 0.1), Inches(y + 0.05),
                                        Inches(bar_w - 0.2), Inches(header_h - 0.1))
        set_text(txH.text_frame.paragraphs[0], phase["header"], Pt(14), WHITE, bold=True)
        txH.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        # ボディ部（下部、LIGHT_GRAY）
        body_y = y + header_h
        body_h = bar_h - header_h
        bdy = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x), Inches(body_y),
            Inches(bar_w), Inches(body_h)
        )
        bdy.fill.solid()
        bdy.fill.fore_color.rgb = LIGHT_GRAY
        bdy.line.color.rgb = BORDER_GRAY
        bdy.line.width = Emu(9525)

        txB = slide.shapes.add_textbox(Inches(x + 0.1), Inches(body_y + 0.1),
                                        Inches(bar_w - 0.2), Inches(body_h - 0.2))
        tf = txB.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        body_text = phase.get("body", "")
        if isinstance(body_text, list):
            for j, item in enumerate(body_text):
                p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                add_rich_runs(p, f"・{item}", base_size=Pt(11), base_color=DARK_GRAY,
                              bold_color=NAVY, line_spacing=1.3)
        else:
            add_rich_runs(tf.paragraphs[0], body_text, base_size=Pt(12),
                          base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.3)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.8 比較スライド（左 vs 右）
def add_compare_slide(prs, title, sub_message, left_title, left_items,
                      right_title, right_items, blank,
                      left_color=ACCENT, right_color=RED_ACCENT,
                      source=None, page_num=None):
    """左右比較スライド

    left_items / right_items: 各3-5項目の短い注釈リスト
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    col_w = 5.7
    left_x = 0.5
    right_x = 6.9
    remaining_h = 6.5 - content_y
    header_h = 0.4

    # 中央 "VS" マーカー
    vs_box = slide.shapes.add_textbox(Inches(6.1), Inches(content_y + 1.5),
                                       Inches(1.0), Inches(0.5))
    p_vs = vs_box.text_frame.paragraphs[0]
    p_vs.alignment = PP_ALIGN.CENTER
    set_text(p_vs, "VS", Pt(18), MEDIUM_GRAY, bold=True)

    # 中央区切り線
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(6.55), Inches(content_y), Emu(9525), Inches(remaining_h)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_GRAY
    line.line.fill.background()

    for side_x, side_title, side_items, side_color in [
        (left_x, left_title, left_items, left_color),
        (right_x, right_title, right_items, right_color),
    ]:
        # カラムヘッダー
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(side_x), Inches(content_y),
            Inches(col_w), Inches(header_h)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = side_color
        bar.line.fill.background()

        txBox = slide.shapes.add_textbox(Inches(side_x + 0.1), Inches(content_y + 0.03),
                                          Inches(col_w - 0.2), Inches(header_h - 0.06))
        set_text(txBox.text_frame.paragraphs[0], side_title, Pt(16), WHITE, bold=True)
        txBox.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        # 注釈（ヘッダー下から下端まで埋める）
        add_annotation_block(slide, side_items, side_x, content_y + header_h + 0.1,
                             col_w, remaining_h - header_h - 0.2, font_size=14)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.9 テーブルスライド
def add_table_slide(prs, title, sub_message, headers, rows, blank,
                    col_widths=None, highlight_rows=None, annotations=None,
                    source=None, page_num=None):
    """テーブル + オプション注釈テキスト

    highlight_rows: ハイライト行のインデックスリスト
    annotations: テーブル横に注釈テキスト表示
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    n_cols = len(headers)
    n_rows = len(rows) + 1

    if annotations:
        table_w = 7.5
        text_x = 8.3
        text_w = 4.5
    else:
        table_w = 12.3
        text_x = None
        text_w = 0

    # 行高を残余スペースに合わせて動的計算
    available_table_h = 6.4 - content_y
    row_h = min(0.55, max(0.35, available_table_h / n_rows))
    table_h = row_h * n_rows

    table_shape = slide.shapes.add_table(
        n_rows, n_cols, Inches(0.5), Inches(content_y), Inches(table_w), Inches(table_h)
    )
    table = table_shape.table
    if col_widths:
        for i, w in enumerate(col_widths):
            table.columns[i].width = Inches(w)

    # Navyヘッダー行
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = ""
        set_text(cell.text_frame.paragraphs[0], header, Pt(13), WHITE, bold=True)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY

    # ゼブラストライプデータ行
    highlight_rows = highlight_rows or []
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i + 1, j)
            cell.text = ""
            set_text(cell.text_frame.paragraphs[0], str(value), Pt(12), DARK_GRAY)
            if i in highlight_rows:
                cell.fill.solid()
                cell.fill.fore_color.rgb = KEY_MSG_BG
            elif i % 2 == 1:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GRAY

    # 注釈テキスト（テーブル横）
    if annotations and text_x:
        remaining_h = 6.4 - content_y
        add_annotation_block(slide, annotations, text_x, content_y, text_w, remaining_h,
                             font_size=13, has_border=True, bg_color=KEY_MSG_BG)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.10 プログレスバースライド
def add_progress_bar_slide(prs, title, sub_message, items, blank,
                           source=None, page_num=None):
    """水平プログレスバー

    items: [{"label":"クラスタA", "value":58, "max_value":100, "color":ACCENT}, ...]
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    n = len(items)
    available_h = 6.5 - content_y
    bar_gap = 0.1
    bar_group_h = (available_h - 0.2) / n
    bar_h = min(0.5, bar_group_h * 0.5)
    label_h = bar_group_h - bar_h - bar_gap
    bar_max_w = 9.0
    colors = [ACCENT, BLUE, NAVY, GREEN_ACCENT, AMBER, RED_ACCENT]

    for i, item in enumerate(items):
        gy = content_y + i * bar_group_h
        color = item.get("color", colors[i % len(colors)])
        pct = item["value"] / max(item.get("max_value", 100), 1)
        bar_w = bar_max_w * pct

        # ラベル（左）
        txL = slide.shapes.add_textbox(Inches(0.5), Inches(gy), Inches(3.0), Inches(label_h))
        set_text(txL.text_frame.paragraphs[0], item["label"], Pt(14), DARK_GRAY, bold=True)

        # 背景バー（グレー、全幅）
        bg_bar = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.5), Inches(gy + label_h),
            Inches(bar_max_w), Inches(bar_h)
        )
        bg_bar.fill.solid()
        bg_bar.fill.fore_color.rgb = LIGHT_GRAY
        bg_bar.line.fill.background()

        # 値バー（色付き）
        if bar_w > 0.1:
            val_bar = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, Inches(3.5), Inches(gy + label_h),
                Inches(bar_w), Inches(bar_h)
            )
            val_bar.fill.solid()
            val_bar.fill.fore_color.rgb = color
            val_bar.line.fill.background()

        # パーセンテージ（バーの右端）
        txP = slide.shapes.add_textbox(Inches(3.5 + bar_w + 0.1), Inches(gy + label_h),
                                        Inches(1.5), Inches(bar_h))
        set_text(txP.text_frame.paragraphs[0],
                 f"{item['value']}{item.get('unit', '%')}", Pt(14), color, bold=True)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.11 トライアングルスライド（3要素の関係図）
def add_triangle_slide(prs, title, sub_message, elements, blank,
                       source=None, page_num=None):
    """3要素トライアングル関係図

    elements: [
        {"title":"技術", "body":"SiC/GaN半導体", "color":NAVY},
        {"title":"市場", "body":"EV・再エネ需要", "color":ACCENT},
        {"title":"政策", "body":"グリーン成長戦略", "color":GREEN_ACCENT},
    ]
    上1 + 下2 の三角配置 + 関係矢印
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    card_w = 3.5
    card_h = 2.0
    colors = [NAVY, ACCENT, GREEN_ACCENT]

    # 三角の3頂点座標
    positions = [
        (5.0, content_y + 0.2),          # 上中央
        (1.5, content_y + 2.8),          # 左下
        (8.5, content_y + 2.8),          # 右下
    ]

    for i, (elem, (px, py)) in enumerate(zip(elements[:3], positions)):
        color = elem.get("color", colors[i % len(colors)])

        # カードヘッダー
        hdr = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(px), Inches(py),
            Inches(card_w), Inches(0.45)
        )
        hdr.fill.solid()
        hdr.fill.fore_color.rgb = color
        hdr.line.fill.background()

        txH = slide.shapes.add_textbox(Inches(px + 0.1), Inches(py + 0.05),
                                        Inches(card_w - 0.2), Inches(0.35))
        set_text(txH.text_frame.paragraphs[0], elem["title"], Pt(14), WHITE, bold=True)
        txH.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

        # カードボディ
        bdy = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(px), Inches(py + 0.45),
            Inches(card_w), Inches(card_h - 0.45)
        )
        bdy.fill.solid()
        bdy.fill.fore_color.rgb = LIGHT_GRAY
        bdy.line.color.rgb = BORDER_GRAY
        bdy.line.width = Emu(9525)

        txB = slide.shapes.add_textbox(Inches(px + 0.15), Inches(py + 0.55),
                                        Inches(card_w - 0.3), Inches(card_h - 0.65))
        tf = txB.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        add_rich_runs(tf.paragraphs[0], elem.get("body", ""),
                      base_size=Pt(12), base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.4)

    # 関係矢印（3辺）
    arrow_pairs = [(0, 1), (0, 2), (1, 2)]
    for a, b in arrow_pairs:
        ax = positions[a][0] + card_w / 2
        ay = positions[a][1] + card_h
        bx = positions[b][0] + card_w / 2
        by = positions[b][1]
        if b == 2 and a == 1:
            ay = positions[a][1] + card_h / 2
            by = positions[b][1] + card_h / 2
        # コネクタ（p:cxnSp）はファイル破損を起こすため、回転矩形で関係線を描く
        _add_line(slide, ax, ay, bx, by, MEDIUM_GRAY, weight_pt=1.0)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.12 ピラミッドスライド
def add_pyramid_slide(prs, title, sub_message, levels, blank,
                      source=None, page_num=None):
    """ピラミッド（上が小、下が大の台形積み重ね）

    levels: [{"title":"萌芽技術", "detail":"ノイズ6テーマ"}, ...]  上→下の順
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    n = len(levels)
    total_h = 6.5 - content_y - 0.2
    level_h = total_h / n
    base_w = 10.0
    center_x = 6.66  # スライド中央
    colors = [RED_ACCENT, AMBER, ACCENT, BLUE, NAVY, RGBColor(0x2E, 0x8B, 0x57)]

    for i, level in enumerate(levels):
        ratio_top = (i + 0.3) / n
        ratio_bot = (i + 1.3) / n
        lw = base_w * (ratio_top + ratio_bot) / 2
        lx = center_x - lw / 2
        ly = content_y + i * level_h
        color = colors[i % len(colors)]

        trap = slide.shapes.add_shape(
            MSO_SHAPE.TRAPEZOID, Inches(lx), Inches(ly),
            Inches(lw), Inches(level_h - 0.05)
        )
        trap.fill.solid()
        trap.fill.fore_color.rgb = color
        trap.line.fill.background()

        txBox = slide.shapes.add_textbox(
            Inches(lx + 0.3), Inches(ly + 0.1),
            Inches(lw - 0.6), Inches(level_h - 0.2)
        )
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        set_text(p, level["title"], Pt(14), WHITE, bold=True)
        if level.get("detail"):
            p2 = tf.add_paragraph()
            p2.alignment = PP_ALIGN.CENTER
            set_text(p2, level["detail"], Pt(11), WHITE)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.13 仮説検証スライド
def add_hypothesis_slide(prs, title, sub_message, hypotheses, blank,
                         source=None, page_num=None):
    """仮説検証テーブル

    hypotheses: [
        {"id":"H1", "hypothesis":"A社は3年以内にシェア首位", "verdict":"partially",
         "evidence":"シェア2位に浮上も首位とのギャップは依然5%"},
        ...
    ]
    verdict: "confirmed" -> OK (緑), "rejected" -> NG (赤),
             "partially" -> 要確認 (黄)
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    headers = ["ID", "仮説", "判定", "エビデンス"]
    n_rows = len(hypotheses) + 1
    available_h = 6.4 - content_y
    row_h = min(0.55, max(0.40, available_h / n_rows))
    table_h = row_h * n_rows

    VERDICT_MAP = {
        "confirmed": ("OK", GREEN_ACCENT),
        "rejected": ("NG", RED_ACCENT),
        "partially": ("---", AMBER),
    }

    table_shape = slide.shapes.add_table(
        n_rows, 4, Inches(0.5), Inches(content_y), Inches(12.3), Inches(table_h)
    )
    table = table_shape.table
    # 列幅: ID=0.8, 仮説=4.5, 判定=1.0, エビデンス=6.0
    table.columns[0].width = Inches(0.8)
    table.columns[1].width = Inches(4.5)
    table.columns[2].width = Inches(1.0)
    table.columns[3].width = Inches(6.0)

    # Navyヘッダー
    for j, header in enumerate(headers):
        cell = table.cell(0, j)
        cell.text = ""
        set_text(cell.text_frame.paragraphs[0], header, Pt(13), WHITE, bold=True)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY

    # データ行
    for i, hyp in enumerate(hypotheses):
        verdict_key = hyp.get("verdict", "partially")
        verdict_label, verdict_color = VERDICT_MAP.get(verdict_key, ("---", AMBER))

        row_data = [hyp.get("id", ""), hyp.get("hypothesis", ""),
                    verdict_label, hyp.get("evidence", "")]

        for j, val in enumerate(row_data):
            cell = table.cell(i + 1, j)
            cell.text = ""
            if j == 2:
                # 判定セルは色付き背景
                set_text(cell.text_frame.paragraphs[0], val, Pt(13), WHITE, bold=True)
                cell.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
                cell.fill.solid()
                cell.fill.fore_color.rgb = verdict_color
            else:
                set_text(cell.text_frame.paragraphs[0], str(val), Pt(12), DARK_GRAY)
                if i % 2 == 1:
                    cell.fill.solid()
                    cell.fill.fore_color.rgb = LIGHT_GRAY

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.14 タイムラインスライド
def add_timeline_slide(prs, title, sub_message, events, blank,
                       source=None, page_num=None):
    """水平タイムライン

    events: [{"year":"2015", "title":"CNF政策支援開始", "color":ACCENT}, ...]
    ラベルは上下交互配置。
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    n = len(events)
    total_w = 11.5
    start_x = 0.9
    line_y = content_y + 1.8  # タイムライン中心位置

    # 水平線
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(start_x), Inches(line_y),
        Inches(total_w), Emu(19050)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = NAVY
    line.line.fill.background()

    step = total_w / max(n - 1, 1) if n > 1 else total_w
    for i, ev in enumerate(events):
        x = start_x + i * step
        color = ev.get("color", ACCENT)

        # マーカー円
        dot = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, Inches(x - 0.12), Inches(line_y - 0.12),
            Inches(0.35), Inches(0.35)
        )
        dot.fill.solid()
        dot.fill.fore_color.rgb = color
        dot.line.fill.background()

        # 年ラベル（マーカー内、白テキスト）
        txY = slide.shapes.add_textbox(Inches(x - 0.25), Inches(line_y - 0.4),
                                        Inches(0.8), Inches(0.25))
        p = txY.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        set_text(p, ev["year"], Pt(10), NAVY, bold=True)

        # イベントテキスト（交互に上下配置して重なり回避）
        if i % 2 == 0:
            ty = line_y + 0.4
        else:
            ty = line_y - 1.0
        txE = slide.shapes.add_textbox(Inches(x - 0.6), Inches(ty),
                                        Inches(1.5), Inches(0.6))
        tf = txE.text_frame
        tf.word_wrap = True
        p2 = tf.paragraphs[0]
        p2.alignment = PP_ALIGN.CENTER
        set_text(p2, ev["title"], Pt(9), DARK_GRAY)

        # 縦線（マーカーからテキストへ）
        if i % 2 == 0:
            vline_y = line_y + 0.2
            vline_h = 0.2
        else:
            vline_y = line_y - 0.5
            vline_h = 0.4
        vline = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(x + 0.03), Inches(vline_y),
            Emu(9525), Inches(vline_h)
        )
        vline.fill.solid()
        vline.fill.fore_color.rgb = color
        vline.line.fill.background()

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 3.15 クロージングスライド
def add_closing_slide(prs, report_title, blank):
    """クロージング — Navy背景 + Thank You"""
    slide = prs.slides.add_slide(blank)
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = NAVY

    # "Thank You"
    txBox = slide.shapes.add_textbox(Inches(1.5), Inches(2.5), Inches(10), Inches(1.5))
    set_text(txBox.text_frame.paragraphs[0], "Thank You", Pt(48), WHITE, bold=True)
    txBox.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # レポートタイトル
    txBox2 = slide.shapes.add_textbox(Inches(1.5), Inches(4.2), Inches(10), Inches(1))
    set_text(txBox2.text_frame.paragraphs[0], report_title, Pt(16), RGBColor(0xAA, 0xBB, 0xDD))
    txBox2.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # APOLLO ブランディング
    txBox3 = slide.shapes.add_textbox(Inches(1.5), Inches(5.5), Inches(10), Inches(0.5))
    set_text(txBox3.text_frame.paragraphs[0], "APOLLO Patent Analytics Platform", Pt(12), RGBColor(0x88, 0x99, 0xBB))
    txBox3.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    # ボトムライン
    bot = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(7.1), Inches(13.33), Emu(27432)
    )
    bot.fill.solid()
    bot.fill.fore_color.rgb = ACCENT
    bot.line.fill.background()
    return slide


# =============================================================================
# 補助スライドタイプ
# =============================================================================

# 目次スライド
def add_toc_slide(prs, title, items, blank, page_num=None):
    """目次スライド — ゼブラストライプ目次

    items = [{"num":1, "title":"セクション名", "page":"P5"}, ...]
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)

    n = len(items)
    table_x, table_y, table_w = 1.5, sub_y + 0.1, 10.3
    row_h = min(0.5, (6.4 - table_y) / max(n, 1))

    for i, item in enumerate(items):
        y = table_y + i * row_h
        if i % 2 == 0:
            bg = slide.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Inches(table_x), Inches(y),
                Inches(table_w), Inches(row_h)
            )
            bg.fill.solid()
            bg.fill.fore_color.rgb = LIGHT_GRAY
            bg.line.fill.background()

        # 番号
        txNum = slide.shapes.add_textbox(Inches(table_x + 0.2), Inches(y + 0.05),
                                          Inches(0.8), Inches(row_h - 0.1))
        set_text(txNum.text_frame.paragraphs[0],
                 f"{item.get('num', i+1)}.", Pt(14), NAVY, bold=True)

        # セクション名
        txTitle = slide.shapes.add_textbox(Inches(table_x + 1.2), Inches(y + 0.05),
                                            Inches(7.0), Inches(row_h - 0.1))
        set_text(txTitle.text_frame.paragraphs[0], item["title"], Pt(14), DARK_GRAY, bold=True)

        # ページ番号
        txPage = slide.shapes.add_textbox(Inches(table_x + 8.5), Inches(y + 0.05),
                                           Inches(1.5), Inches(row_h - 0.1))
        p = txPage.text_frame.paragraphs[0]
        set_text(p, item.get("page", ""), Pt(14), MEDIUM_GRAY)
        p.alignment = PP_ALIGN.RIGHT

    add_bottom_bar_and_footer(slide, page_num)
    return slide


# デュアルパネルスライド（2カラムチャート比較）
def add_dual_panel_slide(prs, title, sub_message,
                          left_label, left_image, left_caption,
                          right_label, right_image, right_caption,
                          left_bullets=None, right_bullets=None,
                          blank=None, source=None, page_num=None):
    """2カラムチャート — 2つの可視化を並列比較"""
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    col_w = 5.9
    left_x = 0.5
    right_x = 6.9
    remaining_h = 6.5 - content_y

    if left_bullets or right_bullets:
        n_bullets = max(len(left_bullets or []), len(right_bullets or []))
        if n_bullets <= 1:
            chart_h = remaining_h * 0.78
        elif n_bullets <= 2:
            chart_h = remaining_h * 0.68
        else:
            chart_h = remaining_h * 0.58
        text_y = content_y + chart_h + 0.1
        text_h = remaining_h - chart_h - 0.3
    else:
        chart_h = remaining_h - 0.5
        text_y = None
        text_h = 0

    for side_x, label, img_path, caption, bullets in [
        (left_x, left_label, left_image, left_caption, left_bullets),
        (right_x, right_label, right_image, right_caption, right_bullets),
    ]:
        add_chart_label(slide, label, side_x, content_y, col_w)
        full_path = os.path.join(SNAP, img_path) if not os.path.isabs(img_path) else img_path
        fit_image(slide, full_path, max_x=side_x, max_y=content_y + 0.3,
                  max_w=col_w, max_h=chart_h - 0.3)

        if caption:
            txBox = slide.shapes.add_textbox(Inches(side_x), Inches(content_y + chart_h),
                                              Inches(col_w), Inches(0.25))
            set_text(txBox.text_frame.paragraphs[0], caption, Pt(9), MEDIUM_GRAY)

        if bullets and text_y:
            add_annotation_block(slide, bullets, side_x, text_y, col_w, text_h, font_size=13)

    # 中央区切り線
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(6.55), Inches(content_y), Emu(9525), Inches(remaining_h)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_GRAY
    line.line.fill.background()

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# ナラティブスライド（テキスト主体 — サマリー/結論限定）
def add_narrative_slide(prs, title, sub_message, paragraphs, blank,
                         source=None, page_num=None):
    """テキスト主体 — エグゼクティブサマリーと結論にのみ使用

    全スライドの10%以下に制限すること。
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y) if sub_message else sub_y + 0.1

    remaining_h = 6.5 - content_y
    add_annotation_block(slide, paragraphs, 0.5, content_y, 12.3, remaining_h, font_size=16)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# チャート全画面スライド
def add_image_slide(prs, title, sub_message, image_path, blank,
                    caption=None, chart_label=None, source=None, page_num=None):
    """チャート全画面 — 画像が主役のスライド"""
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    if chart_label:
        add_chart_label(slide, chart_label, 0.5, content_y, 12.3)
        img_y = content_y + 0.3
    else:
        img_y = content_y

    full_path = os.path.join(SNAP, image_path) if not os.path.isabs(image_path) else image_path
    fit_image(slide, full_path, max_x=0.5, max_y=img_y, max_w=12.3, max_h=6.4 - img_y)

    if caption:
        txBox = slide.shapes.add_textbox(Inches(0.5), Inches(6.45), Inches(12.3), Inches(0.25))
        set_text(txBox.text_frame.paragraphs[0], caption, Pt(10), MEDIUM_GRAY)
        txBox.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 推奨アクションスライド
def add_recommendation_slide(prs, title, sub_message, recommendations, blank,
                              source=None, page_num=None):
    """推奨アクション — 優先度バー付き

    recommendations: [{"priority":"高","title":"出願強化","timeframe":"短期","desc":"..."},...]
    """
    PRIORITY_COLORS = {"高": RED_ACCENT, "中": AMBER, "低": GREEN_ACCENT}
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    n = len(recommendations)
    available_h = 6.5 - content_y
    card_h = min(1.3, (available_h - 0.1 * (n - 1)) / n)

    for i, rec in enumerate(recommendations):
        y = content_y + i * (card_h + 0.1)
        p_color = PRIORITY_COLORS.get(rec.get("priority", "中"), AMBER)

        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(y), Emu(54864), Inches(card_h)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = p_color
        bar.line.fill.background()

        card = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(y), Inches(12.0), Inches(card_h)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_GRAY
        card.line.color.rgb = BORDER_GRAY
        card.line.width = Emu(9525)

        txBox_p = slide.shapes.add_textbox(Inches(1.0), Inches(y + 0.08), Inches(0.8), Inches(0.3))
        set_text(txBox_p.text_frame.paragraphs[0], f"[{rec.get('priority', '中')}]", Pt(10), p_color, bold=True)

        txBox_t = slide.shapes.add_textbox(Inches(1.9), Inches(y + 0.08), Inches(5.5), Inches(0.3))
        set_text(txBox_t.text_frame.paragraphs[0], rec["title"], Pt(16), NAVY, bold=True)

        if rec.get("timeframe"):
            txBox_tf = slide.shapes.add_textbox(Inches(9.0), Inches(y + 0.08), Inches(3.5), Inches(0.3))
            p_tf = txBox_tf.text_frame.paragraphs[0]
            set_text(p_tf, rec["timeframe"], Pt(13), MEDIUM_GRAY)
            p_tf.alignment = PP_ALIGN.RIGHT

        if rec.get("desc"):
            txBox_d = slide.shapes.add_textbox(Inches(1.9), Inches(y + 0.4), Inches(10.5), Inches(card_h - 0.5))
            tf_d = txBox_d.text_frame
            tf_d.word_wrap = True
            tf_d.auto_size = MSO_AUTO_SIZE.NONE
            add_rich_runs(tf_d.paragraphs[0], rec["desc"], base_size=Pt(13),
                          base_color=DARK_GRAY, bold_color=NAVY)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# =============================================================================
# 章構成スライド（参考デッキ準拠・主張骨格 §0.9-A0 を体現）
# =============================================================================

# 重心移動スライド（PAST → PRESENT・エグゼクティブサマリー型）
def add_shift_slide(prs, title, lead, past, present, closing, blank,
                    eyebrow=None, source=None, page_num=None):
    """重心移動（PAST → PRESENT）— 過去の主役から現在の重点への移行を1枚で語る。

    左カード「PAST ・ 過去の主役」、中央に NAVY の右向き矢印、右カード
    「PRESENT ・ 現在の重点」を同サイズで並べ、下部に頑健性の締め文を置く。

    Args:
        title: 主張見出し（結論性のある名詞句）
        lead: リード文（タイトル直下の■サブメッセージ。`add_sub_message` 使用）
        past / present: dict `{"label":"PAST ・ 過去の主役", "heading":"短い名詞句",
                              "desc":"1〜2文"}`
        closing: 締め文（下部に地の文一文。「4つの独立した手法がいずれも同じ
                 方向を指す、頑健な結論」等）
        eyebrow: タイトル直上のアイブロウ（章/モジュール名）
    主張骨格（§0.9-A0）: アイブロウ→主張見出し→リード文→根拠（PAST/PRESENT 2枚）
                        →締め文（So What）。
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title, eyebrow=eyebrow)
    content_y = add_sub_message(slide, lead, y=sub_y)

    # 締め文を下部に確保し、その上にカード帯を置く
    closing_h = 0.75 if closing else 0.0
    band_top = content_y + 0.10
    band_bottom = 6.45 - (closing_h + 0.15 if closing else 0.0)
    band_h = band_bottom - band_top

    arrow_w = 1.05
    gap = 0.30
    total_w = 12.3
    card_w = (total_w - arrow_w - gap * 2) / 2
    left_x = 0.5
    arrow_x = left_x + card_w + gap
    right_x = arrow_x + arrow_w + gap

    for cx, data, accent in [(left_x, past, MEDIUM_GRAY), (right_x, present, ACCENT)]:
        card = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(cx), Inches(band_top),
            Inches(card_w), Inches(band_h)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_GRAY
        card.line.color.rgb = BORDER_GRAY
        card.line.width = Emu(9525)
        # 上端アクセント帯（PAST=ミュート／PRESENT=ACCENT で現在を際立たせる）
        topbar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(cx), Inches(band_top),
            Inches(card_w), Emu(54864)
        )
        topbar.fill.solid()
        topbar.fill.fore_color.rgb = accent
        topbar.line.fill.background()

        tb = slide.shapes.add_textbox(Inches(cx + 0.25), Inches(band_top + 0.18),
                                      Inches(card_w - 0.50), Inches(band_h - 0.36))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE   # 上下中央（スカスカ防止・§0.9-G）
        # label（小・ミュート）
        set_text(tf.paragraphs[0], data.get("label", ""), Pt(11), MEDIUM_GRAY, weight="medium")
        # heading（Black Navy）
        ph = tf.add_paragraph()
        ph.space_before = Pt(4)
        set_text(ph, data.get("heading", ""), Pt(20), NAVY, weight="black", line_spacing=1.15)
        # desc（Regular）
        if data.get("desc"):
            pd = tf.add_paragraph()
            pd.space_before = Pt(6)
            add_rich_runs(pd, data["desc"], base_size=Pt(13),
                          base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.35)

    # 中央の右向き矢印（NAVY・オートシェイプ。コネクタ禁止）
    arr_h = 0.70
    arr = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, Inches(arrow_x), Inches(band_top + band_h / 2 - arr_h / 2),
        Inches(arrow_w), Inches(arr_h)
    )
    arr.fill.solid()
    arr.fill.fore_color.rgb = NAVY
    arr.line.fill.background()

    # 締め文（下部・地の文一文）
    if closing:
        cb = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(band_bottom + 0.15),
            Inches(12.3), Inches(closing_h)
        )
        cb.fill.solid()
        cb.fill.fore_color.rgb = KEY_MSG_BG
        cb.line.fill.background()
        cbar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(band_bottom + 0.15),
            Emu(36576), Inches(closing_h)
        )
        cbar.fill.solid()
        cbar.fill.fore_color.rgb = NAVY
        cbar.line.fill.background()
        ctb = slide.shapes.add_textbox(Inches(0.78), Inches(band_bottom + 0.15),
                                       Inches(11.9), Inches(closing_h))
        ctf = ctb.text_frame
        ctf.word_wrap = True
        ctf.auto_size = MSO_AUTO_SIZE.NONE
        ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
        add_rich_runs(ctf.paragraphs[0], closing, base_size=Pt(14),
                      base_color=NAVY, bold_color=NAVY, line_spacing=1.35, weight="medium")

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# クロス統合スライド（N手法が同じ結論へ収束）
def add_convergence_slide(prs, title, methods, conclusion, blank,
                          eyebrow=None, source=None, page_num=None):
    """クロス統合（N手法 → 1つの頑健な結論へ収束）。

    左に N 個の手法行（手法名→その手法の発見を1文）、各行から細い矢印が
    右の大きな「頑健な結論」ボックスへ収束する。

    Args:
        methods: list of `{"method":"俯瞰図分析", "finding":"…1文"}`（3〜5件）
        conclusion: dict `{"headline":"…結論見出し", "detail":"…手法に依存しない構造変化"}`
        eyebrow: タイトル直上のアイブロウ
    主張骨格（§0.9-A0）: 複数手法を束ねた「束ねスライド」（§0.9-C）。各手法の発見＝
                        根拠、右の結論箱＝主張見出し＋So What。
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title, eyebrow=eyebrow)
    # リード文は結論見出しを流用して主張骨格を満たす
    content_y = add_sub_message(slide, conclusion.get("headline", ""), y=sub_y)

    area_top = content_y + 0.15
    area_bottom = 6.45
    area_h = area_bottom - area_top

    left_x = 0.5
    method_w = 5.6
    arrow_w = 0.55
    concl_x = left_x + method_w + arrow_w + 0.35
    concl_w = 12.83 - concl_x

    n = len(methods)
    gap = 0.18
    row_h = min(1.30, (area_h - gap * (n - 1)) / max(n, 1))
    total_rows_h = row_h * n + gap * (n - 1)
    start_y = area_top + (area_h - total_rows_h) / 2
    method_colors = [NAVY, BLUE, ACCENT, GREEN_ACCENT, AMBER]

    row_centers = []
    for i, m in enumerate(methods):
        ry = start_y + i * (row_h + gap)
        row_centers.append(ry + row_h / 2)
        color = method_colors[i % len(method_colors)]
        # 手法行（左に色帯＋手法名＋発見文）
        box = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(left_x), Inches(ry),
            Inches(method_w), Inches(row_h)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT_GRAY
        box.line.color.rgb = BORDER_GRAY
        box.line.width = Emu(9525)
        bar = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(left_x), Inches(ry), Inches(0.12), Inches(row_h)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()
        tb = slide.shapes.add_textbox(Inches(left_x + 0.28), Inches(ry + 0.08),
                                      Inches(method_w - 0.45), Inches(row_h - 0.16))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        set_text(tf.paragraphs[0], m.get("method", ""), Pt(13), NAVY, weight="medium")
        if m.get("finding"):
            pf = tf.add_paragraph()
            pf.space_before = Pt(2)
            add_rich_runs(pf, m["finding"], base_size=Pt(11),
                          base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.25)

    # 結論ボックス（ACCENT 上端帯＋KEY_MSG_BG 背景）
    cb = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(concl_x), Inches(area_top),
        Inches(concl_w), Inches(area_h)
    )
    cb.fill.solid()
    cb.fill.fore_color.rgb = KEY_MSG_BG
    cb.line.color.rgb = ACCENT
    cb.line.width = Emu(12700)
    topbar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(concl_x), Inches(area_top), Inches(concl_w), Emu(64008)
    )
    topbar.fill.solid()
    topbar.fill.fore_color.rgb = ACCENT
    topbar.line.fill.background()
    ctb = slide.shapes.add_textbox(Inches(concl_x + 0.28), Inches(area_top + 0.25),
                                   Inches(concl_w - 0.56), Inches(area_h - 0.50))
    ctf = ctb.text_frame
    ctf.word_wrap = True
    ctf.auto_size = MSO_AUTO_SIZE.NONE
    ctf.vertical_anchor = MSO_ANCHOR.MIDDLE
    # ミニ見出し（収束の宣言）
    set_text(ctf.paragraphs[0], "頑健な結論", Pt(11), MEDIUM_GRAY, weight="medium")
    ph = ctf.add_paragraph()
    ph.space_before = Pt(6)
    add_rich_runs(ph, conclusion.get("headline", ""), base_size=Pt(19),
                  base_color=NAVY, bold_color=NAVY, line_spacing=1.2, weight="black")
    if conclusion.get("detail"):
        pdt = ctf.add_paragraph()
        pdt.space_before = Pt(8)
        add_rich_runs(pdt, conclusion["detail"], base_size=Pt(13),
                      base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.4)

    # 各手法行→結論箱への細い矢印（オートシェイプ。コネクタ禁止）
    concl_mid_y = area_top + area_h / 2
    arr_x = left_x + method_w + 0.06
    for cyc in row_centers:
        _add_line(slide, arr_x, cyc, concl_x - 0.04, concl_mid_y, ACCENT, weight_pt=1.2)
    # 収束先を示す右向き矢印（結論箱直前）
    head = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW, Inches(concl_x - arrow_w - 0.02),
        Inches(concl_mid_y - 0.13), Inches(arrow_w), Inches(0.26)
    )
    head.fill.solid()
    head.fill.fore_color.rgb = ACCENT
    head.line.fill.background()

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 優先度別アクションスライド（優先度バッジ＋期間ピル）
def add_priority_actions_slide(prs, title, actions, blank,
                               eyebrow=None, source=None, page_num=None):
    """優先度別アクション — 各行に [優先度バッジ 高/中/低] ＋ 見出し ＋ 詳細 ＋ [期間ピル]。

    既存 `add_recommendation_slide` の参考デッキ準拠版。優先度を色バッジ（ピル）、
    期間を小さめのピルで明示し、行単位で読み取りやすくする。
    （優先度＋期間＋詳細だけで足りる場合は `add_recommendation_slide` でも可。）

    Args:
        actions: list of `{"priority":"高"/"中"/"低", "title":"…", "detail":"…1文",
                          "timeframe":"短期・1年以内"}`
        eyebrow: タイトル直上のアイブロウ
    優先度色: 高=RED_ACCENT、中=AMBER、低=MEDIUM_GRAY。
    """
    PRIORITY_COLORS = {"高": RED_ACCENT, "中": AMBER, "低": MEDIUM_GRAY}
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title, eyebrow=eyebrow)
    content_y = sub_y + 0.10

    n = len(actions)
    available_h = 6.45 - content_y
    gap = 0.14
    card_h = min(1.35, (available_h - gap * (n - 1)) / max(n, 1))

    badge_w = 0.95
    badge_x = 0.5
    text_x = badge_x + badge_w + 0.30
    pill_w = 2.40
    pill_x = 12.83 - pill_w
    text_w = pill_x - text_x - 0.25

    for i, act in enumerate(actions):
        y = content_y + i * (card_h + gap)
        p_color = PRIORITY_COLORS.get(act.get("priority", "中"), AMBER)

        # カード地
        card = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(y), Inches(12.33), Inches(card_h)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = LIGHT_GRAY
        card.line.color.rgb = BORDER_GRAY
        card.line.width = Emu(9525)

        # 優先度バッジ（色付きピル・白文字）
        badge = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(badge_x + 0.12),
            Inches(y + card_h / 2 - 0.26), Inches(badge_w), Inches(0.52)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = p_color
        badge.line.fill.background()
        btf = badge.text_frame
        btf.word_wrap = True
        btf.margin_top = Emu(0)
        btf.margin_bottom = Emu(0)
        btf.vertical_anchor = MSO_ANCHOR.MIDDLE
        bp = btf.paragraphs[0]
        bp.alignment = PP_ALIGN.CENTER
        set_text(bp, act.get("priority", "中"), Pt(15), WHITE, weight="black")

        # 見出し（Navy・太字）
        tb = slide.shapes.add_textbox(Inches(text_x), Inches(y + 0.10),
                                      Inches(text_w), Inches(card_h - 0.18))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        set_text(tf.paragraphs[0], act.get("title", ""), Pt(15), NAVY, bold=True)
        if act.get("detail"):
            pdc = tf.add_paragraph()
            pdc.space_before = Pt(3)
            add_rich_runs(pdc, act["detail"], base_size=Pt(12),
                          base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.3)

        # 期間ピル（小さめ・枠線）
        if act.get("timeframe"):
            pill = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE, Inches(pill_x),
                Inches(y + card_h / 2 - 0.20), Inches(pill_w - 0.15), Inches(0.40)
            )
            pill.fill.solid()
            pill.fill.fore_color.rgb = WHITE
            pill.line.color.rgb = p_color
            pill.line.width = Emu(12700)
            ptf = pill.text_frame
            ptf.word_wrap = True
            ptf.margin_top = Emu(0)
            ptf.margin_bottom = Emu(0)
            ptf.vertical_anchor = MSO_ANCHOR.MIDDLE
            pp = ptf.paragraphs[0]
            pp.alignment = PP_ALIGN.CENTER
            set_text(pp, act["timeframe"], Pt(11), p_color, weight="medium")

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# アクションアイテムスライド（☐チェックリスト・締め寄り）
def add_action_items_slide(prs, title, items, blank,
                           eyebrow=None, brand_line=None, source=None, page_num=None):
    """アクションアイテム（☐ チェックリスト）— 完結したアクション文を縦に並べる。

    各行に ☐（関数側で付与）＋ アクション文。下部に任意のブランド行を置き、
    結論・締め寄りの体裁にする。

    Args:
        items: list of str（各「完結したアクション文」。☐は関数が付与）
        brand_line: 省略可（例「APOLLO ・ 特許ランドスケープ分析」）。締め寄りのタグライン。
        eyebrow: タイトル直上のアイブロウ
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title, eyebrow=eyebrow)
    content_y = sub_y + 0.10

    brand_h = 0.55 if brand_line else 0.0
    area_bottom = 6.45 - (brand_h + 0.15 if brand_line else 0.0)
    available_h = area_bottom - content_y

    n = len(items)
    gap = 0.14
    row_h = min(1.0, (available_h - gap * (n - 1)) / max(n, 1))
    total_h = row_h * n + gap * (n - 1)
    start_y = content_y + (available_h - total_h) / 2  # 縦中央寄せ

    box_x = 0.5
    box_w = 12.33
    check_w = 0.42

    for i, item in enumerate(items):
        y = start_y + i * (row_h + gap)
        row = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(box_x), Inches(y), Inches(box_w), Inches(row_h)
        )
        row.fill.solid()
        row.fill.fore_color.rgb = LIGHT_GRAY if i % 2 == 0 else WHITE
        row.line.color.rgb = BORDER_GRAY
        row.line.width = Emu(9525)

        # ☐ チェックボックス（角丸枠・ACCENT 線）
        chk = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, Inches(box_x + 0.28),
            Inches(y + row_h / 2 - check_w / 2), Inches(check_w), Inches(check_w)
        )
        chk.fill.solid()
        chk.fill.fore_color.rgb = WHITE
        chk.line.color.rgb = ACCENT
        chk.line.width = Emu(19050)

        # アクション文
        tb = slide.shapes.add_textbox(Inches(box_x + 0.28 + check_w + 0.22),
                                      Inches(y + 0.06),
                                      Inches(box_w - check_w - 0.85), Inches(row_h - 0.12))
        tf = tb.text_frame
        tf.word_wrap = True
        tf.auto_size = MSO_AUTO_SIZE.NONE
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        add_rich_runs(tf.paragraphs[0], item, base_size=Pt(14),
                      base_color=DARK_GRAY, bold_color=NAVY, line_spacing=1.3)

    # ブランド行（締め寄り）
    if brand_line:
        bb = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(area_bottom + 0.15),
            Inches(12.33), Inches(brand_h)
        )
        bb.fill.solid()
        bb.fill.fore_color.rgb = NAVY
        bb.line.fill.background()
        btf = bb.text_frame
        btf.word_wrap = True
        btf.vertical_anchor = MSO_ANCHOR.MIDDLE
        bp = btf.paragraphs[0]
        bp.alignment = PP_ALIGN.CENTER
        set_text(bp, brand_line, Pt(13), WHITE, weight="medium")

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide


# 2x2マトリクススライド（旧式・軸ラベルのみ）
def add_matrix_slide(prs, title, sub_message, quadrants, blank,
                     x_label="→ 成長率", y_label="↑ 規模",
                     source=None, page_num=None):
    """2x2マトリクス（4象限）

    quadrants: {"TL":{"title":"新興","items":["A社"]}, "TR":..., "BL":..., "BR":...}
    """
    slide = prs.slides.add_slide(blank)
    sub_y = add_title_shape(slide, title)
    content_y = add_sub_message(slide, sub_message, y=sub_y)

    mx = 1.5
    my = content_y + 0.2
    mw = 5.5
    mh = 6.0 - content_y
    half_w = mw / 2
    half_h = mh / 2
    quad_colors = {"TL": ACCENT, "TR": GREEN_ACCENT, "BL": MEDIUM_GRAY, "BR": RED_ACCENT}
    positions = {"TL": (mx, my), "TR": (mx + half_w, my),
                 "BL": (mx, my + half_h), "BR": (mx + half_w, my + half_h)}

    for key, pos in positions.items():
        q = quadrants.get(key, {})
        color = quad_colors[key]
        qx, qy = pos

        box = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, Inches(qx), Inches(qy),
            Inches(half_w - 0.05), Inches(half_h - 0.05)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = LIGHT_GRAY
        box.line.color.rgb = BORDER_GRAY
        box.line.width = Emu(9525)

        txBox = slide.shapes.add_textbox(
            Inches(qx + 0.1), Inches(qy + 0.1),
            Inches(half_w - 0.3), Inches(0.35)
        )
        set_text(txBox.text_frame.paragraphs[0], q.get("title", ""), Pt(13), color, bold=True)

        items = q.get("items", [])
        if items:
            txBox2 = slide.shapes.add_textbox(
                Inches(qx + 0.15), Inches(qy + 0.5),
                Inches(half_w - 0.4), Inches(half_h - 0.7)
            )
            tf = txBox2.text_frame
            tf.word_wrap = True
            for j, item in enumerate(items[:5]):
                p = tf.paragraphs[0] if j == 0 else tf.add_paragraph()
                set_text(p, f"・{item}", Pt(10), DARK_GRAY)

    # 右側に注釈用スペース（マトリクスの解説）
    ann_x = mx + mw + 0.5
    ann_w = 13.33 - ann_x - 0.5
    if ann_w > 2.0:
        txAnn = slide.shapes.add_textbox(Inches(ann_x), Inches(my),
                                          Inches(ann_w), Inches(mh))
        tf_ann = txAnn.text_frame
        tf_ann.word_wrap = True

    # 軸ラベル
    txX = slide.shapes.add_textbox(Inches(mx + mw/2 - 0.5), Inches(my + mh + 0.05),
                                    Inches(1.5), Inches(0.25))
    set_text(txX.text_frame.paragraphs[0], x_label, Pt(10), MEDIUM_GRAY)
    txY = slide.shapes.add_textbox(Inches(mx - 0.6), Inches(my + mh/2 - 0.15),
                                    Inches(0.5), Inches(0.3))
    set_text(txY.text_frame.paragraphs[0], y_label, Pt(10), MEDIUM_GRAY)

    if source:
        add_source_label(slide, source)
    add_bottom_bar_and_footer(slide, page_num)
    return slide
