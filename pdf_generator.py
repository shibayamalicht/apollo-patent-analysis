import io
import os
import re
import platform
import utils
import streamlit as st
from PIL import Image

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, PageBreak, Table, TableStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.units import mm
    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False

class ReportGenerator:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.font_name = "Helvetica" # Fallback
        self.setup_fonts()
        self.setup_styles()

    def setup_fonts(self):
        if not HAS_REPORTLAB: return
        
        # --- Strategy: OS-Native (Mac) > Downloaded (Noto) > Fallback ---
        
        # 1. Mac OS Native (Highest Quality)
        if platform.system() == "Darwin":
            # Hiragino Sans is standard on macOS
            mac_font = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
            if not os.path.exists(mac_font):
                 # Fallback to older name
                 mac_font = "/Library/Fonts/Hiragino Sans W3.ttc"
            
            if os.path.exists(mac_font):
                try:
                    pdfmetrics.registerFont(TTFont('HiraginoSans', mac_font, subfontIndex=0))
                    self.font_name = 'HiraginoSans'
                    return
                except: pass

        # 2. Universal: Noto Sans JP (Downloaded)
        font_dir = "fonts"
        if not os.path.exists(font_dir):
            os.makedirs(font_dir)
            
        target_font_path = os.path.join(font_dir, "NotoSansJP-Regular.ttf")
        font_url = "https://raw.githubusercontent.com/minoryorg/noto-sans-jp/master/fonts/NotoSansJP-Regular.ttf"
        
        if not os.path.exists(target_font_path):
            try:
                import requests
                response = requests.get(font_url, timeout=15)
                if response.status_code == 200:
                    with open(target_font_path, 'wb') as f:
                        f.write(response.content)
            except Exception: pass

        if os.path.exists(target_font_path):
            try:
                pdfmetrics.registerFont(TTFont('NotoSansJP', target_font_path))
                self.font_name = 'NotoSansJP'
                return
            except: pass
        
        # 3. Fallback: IPAex
        fallback_path = "ipaexg.ttf"
        if os.path.exists(fallback_path):
            try:
                pdfmetrics.registerFont(TTFont('IPAexGothic', fallback_path))
                self.font_name = 'IPAexGothic'
                return
            except: pass

        # 4. Final Fallback: Heisei
        self.font_name = "HeiseiKakuGo-W5"
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))

    def setup_styles(self):
        if not HAS_REPORTLAB: return
        from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
        
        color_primary = colors.HexColor('#0B1E3B')
        color_accent = colors.HexColor('#81D4FA')
        color_text = colors.HexColor('#333333')
        
        self.styles = getSampleStyleSheet()
        for style_name in self.styles.byName:
            style = self.styles[style_name]
            style.fontName = self.font_name
            style.textColor = color_text

        # 1. Executive Summary Box
        self.styles.add(ParagraphStyle(
            name='ExecSummaryBody',
            parent=self.styles['Normal'],
            fontName=self.font_name,
            fontSize=10.5,
            leading=16,
            textColor=color_primary,
            backColor=colors.HexColor('#EBF5FB'),
            borderPadding=(10, 10, 10, 10),
            borderRadius=4,
            spaceBefore=10,
            spaceAfter=20,
            alignment=TA_JUSTIFY
        ))

        # 2. H1
        self.styles.add(ParagraphStyle(
            name='ReportHeading1',
            parent=self.styles['Heading1'],
            fontName=self.font_name,
            fontSize=18,
            leading=24,
            spaceBefore=24,
            spaceAfter=12,
            textColor=color_primary,
            borderColor=color_accent,
            borderWidth=0,
            alignment=TA_LEFT
        ))

        # 3. H2
        self.styles.add(ParagraphStyle(
            name='ReportHeading2',
            parent=self.styles['Heading2'],
            fontName=self.font_name,
            fontSize=13,
            leading=18,
            spaceBefore=18,
            spaceAfter=8,
            textColor=color_primary,
            textTransform='uppercase',
            keepWithNext=True
        ))
        
        # 3.5 H3
        self.styles.add(ParagraphStyle(
            name='ReportHeading3',
            parent=self.styles['Heading3'],
            fontName=self.font_name,
            fontSize=11,
            leading=14,
            spaceBefore=12,
            spaceAfter=6,
            textColor=color_primary,
            keepWithNext=True
        ))

        # 4. Normal
        self.styles.add(ParagraphStyle(
            name='ReportNormal',
            parent=self.styles['Normal'],
            fontName=self.font_name,
            fontSize=10.5,
            leading=17,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            textColor=color_text
        ))

        # 4.5 Bullet
        self.styles.add(ParagraphStyle(
            name='ReportBullet',
            parent=self.styles['Normal'],
            fontName=self.font_name,
            fontSize=10.5,
            leading=17,
            leftIndent=15,
            firstLineIndent=0,
            bulletIndent=5,
            spaceAfter=8,
            alignment=TA_LEFT,
            textColor=color_text
        ))

        # 5. Caption
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Normal'],
            fontName=self.font_name,
            fontSize=9,
            alignment=TA_CENTER,
            textColor=colors.gray,
            spaceBefore=4,
            spaceAfter=12
        ))

        # 6. Title Page
        self.styles.add(ParagraphStyle(
            name='TitleBig',
            fontName=self.font_name,
            fontSize=36,
            leading=44,
            textColor=colors.white,
            alignment=TA_LEFT,
        ))
        self.styles.add(ParagraphStyle(
            name='TitleSmall',
            fontName=self.font_name,
            fontSize=14,
            leading=20,
            textColor=color_accent,
            alignment=TA_LEFT,
        ))

    def parse_markdown(self, text, snapshots):
        flowables = []
        def replace_bold(match): return f"<b>{match.group(1)}</b>"
        
        lines = text.split('\n')
        in_exec_summary = False
        
        for line in lines:
            line = line.strip()
            if not line:
                flowables.append(Spacer(1, 3*mm))
                continue
            
            # --- Box Logic Check ---
            # Enter box
            if "# 1. Executive Strategy Brief" in line or "# 1. Executive Insight" in line:
                 in_exec_summary = True
            # EXIT Box: If any header is detected that is NOT the box title
            elif line.startswith("#"):
                 # Simple check: If it starts with #, and it's not the entry line (already passed), exit.
                 # Since we process line by line, if we hit a NEW header, we exit.
                 in_exec_summary = False
            
            # HR
            if re.match(r'^[-*_]{3,}$', line):
                 flowables.append(Spacer(1, 5*mm))
                 continue

            # --- Image + Text Logic ---
            # Regex: [[Evidence 1, 2]] or [Evidence 1] (Supports multiple and single brackets)
            evidence_pattern = r'\[{1,2}Evidence\s+([0-9,\s-]+)\]{1,2}'
            
            # 1. Collect Images from Line
            line_images = []
            matches = re.finditer(evidence_pattern, line, flags=re.IGNORECASE)
            for m in matches:
                ids_str = m.group(1)
                # Split by comma to handle [[Evidence 1, 2]]
                raw_ids = [x.strip() for x in ids_str.split(',')]
                for raw_id in raw_ids:
                    # Parse 1 or 1-2
                    if '-' in raw_id:
                        parts = raw_id.split('-')
                        e_num = parts[0]
                        e_sub = parts[1] if len(parts)>1 else None
                    else:
                        e_num = raw_id
                        e_sub = None
                    
                    if e_num and e_num.isdigit():
                         line_images.append((e_num, e_sub))
            
            # 2. Text Processing: Replace tags with styled text
            def replace_ev_tag_multi(match):
                ids_str = match.group(1)
                raw_ids = [x.strip() for x in ids_str.split(',')]
                styled_tags = []
                for raw_id in raw_ids:
                     styled_tags.append(f'<b><font color="#0B1E3B">[Evidence {raw_id}]</font></b>')
                return " ".join(styled_tags)
            
            proc_line = re.sub(evidence_pattern, replace_ev_tag_multi, line, flags=re.IGNORECASE)
            proc_line = re.sub(r'\*\*(.*?)\*\*', replace_bold, proc_line)
            
            style = self.styles['ReportNormal']
            
            # Header Handling (Safe)
            if line.startswith('# '):
                 if flowables: flowables.append(PageBreak())
                 style = self.styles['ReportHeading1']
                 proc_line = re.sub(r'^#\s+', '', proc_line)
            elif line.startswith('## '):
                 style = self.styles['ReportHeading2']
                 proc_line = re.sub(r'^##\s+', '', proc_line)
            elif line.startswith('### '):
                 style = self.styles['ReportHeading3']
                 proc_line = re.sub(r'^###\s+', '', proc_line)
            elif line.startswith('#### '):
                 # Handle H4 as Bold Normal
                 style = self.styles['ReportNormal']
                 temp_text = re.sub(r'^####\s+', '', proc_line)
                 proc_line = f"<b>{temp_text}</b>"
            elif line.startswith('- ') or line.startswith('* '):
                 style = self.styles['ReportBullet']
                 proc_line = '• ' + proc_line[2:]
            
            # Apply Box Style if applicable
            if in_exec_summary and not line.startswith('#'):
                 style = self.styles['ExecSummaryBody']

            flowables.append(Paragraph(proc_line, style))

            # 3. Append Images
            if line_images:
                seen_evs = set()
                for (ev_num_str, sub_idx_str) in line_images:
                    
                    full_tag = f"{ev_num_str}-{sub_idx_str}" if sub_idx_str else ev_num_str
                    if full_tag in seen_evs: continue
                    seen_evs.add(full_tag)
                    
                    ev_id = int(ev_num_str) - 1
                    
                    if 0 <= ev_id < len(snapshots):
                        snap = snapshots[ev_id]
                        
                        target_img_bytes = None
                        
                        # Logic: 
                        # If sub_index provided (e.g. 1-2) -> Get image at index 1 (0-based) from 'images' list
                        # If no sub_index -> Get 'image' (main)
                        
                        if sub_idx_str:
                            sub_id = int(sub_idx_str) - 1
                            if 'images' in snap and snap['images'] and 0 <= sub_id < len(snap['images']):
                                target_img_bytes = snap['images'][sub_id]
                        else:
                            target_img_bytes = snap.get('image')

                        if target_img_bytes:
                            try:
                                img = Image.open(io.BytesIO(target_img_bytes))
                                w, h = img.size
                                aspect = h / float(w)
                                disp_w = 160*mm
                                disp_h = disp_w * aspect
                                if disp_h > 120*mm: disp_h = 120*mm; disp_w = disp_h / aspect
                                rl_img = RLImage(io.BytesIO(target_img_bytes), width=disp_w, height=disp_h)
                                flowables.append(Spacer(1, 3*mm))
                                flowables.append(rl_img)
                                
                                # Caption
                                cap_suffix = f" ({sub_idx_str})" if sub_idx_str else ""
                                flowables.append(Paragraph(f"<b>Fig.{ev_id+1}{cap_suffix}</b>: {snap['title']}", self.styles['Caption']))
                                flowables.append(Spacer(1, 3*mm))
                            except: pass
            
            continue
            
        return flowables

    def draw_cover(self, canvas, doc):
        """Deep Space Cover Page"""
        canvas.saveState()
        w, h = A4
        
        # 1. Background: Midnight Blue
        # canvas.setFillColor(colors.HexColor('#0B1E3B'))
        # canvas.rect(0, 0, w, h, fill=1, stroke=0)
        
        # Design B: White Logic with Massive Blue Header
        # Header Block
        canvas.setFillColor(colors.HexColor('#0B1E3B'))
        canvas.rect(0, h * 0.4, w, h * 0.6, fill=1, stroke=0) # Top 60% is blue
        
        # Accent Line (Ice Blue)
        canvas.setStrokeColor(colors.HexColor('#81D4FA'))
        canvas.setLineWidth(4)
        canvas.line(20*mm, h * 0.4, 190*mm, h * 0.4) # Dividing line
        
        # Titles (Center them safely)
        # We handle titles in flowables usually, but here manual placement gives control
        # Text is white on blue
        canvas.setFont(self.font_name, 32)
        canvas.setFillColor(colors.white)
        # canvas.drawString(30*mm, h * 0.7, "MARKET INTELLIGENCE")
        
        # Footer Area (White bottom)
        canvas.setFont(self.font_name, 10)
        canvas.setFillColor(colors.HexColor('#333333'))
        canvas.drawCentredString(w/2, 20*mm, "CONFIDENTIAL STRATEGIC REPORT • GENERATED BY VOYAGER")
        
        canvas.restoreState()

    def create_pdf(self, report_text, snapshots, mission_objective="", subtitle=""):
        if not HAS_REPORTLAB: return None, "No ReportLab"
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=25*mm, leftMargin=25*mm,
            topMargin=25*mm, bottomMargin=25*mm
        )
        
        story = []
        
        # --- TITLE PAGE CONTENT (In Flowables for easier text wrapping) ---
        story.append(Spacer(1, 60*mm))
        story.append(Paragraph("INTELLIGENCE REPORT", self.styles['TitleBig']))
        story.append(Spacer(1, 10*mm))
        if subtitle:
            story.append(Paragraph(subtitle, self.styles['TitleSmall']))
        else:
            story.append(Paragraph("MARKET INTELLIGENCE & IP INTELLIGENCE", self.styles['TitleSmall']))
        
        import datetime
        d = datetime.datetime.now().strftime("%Y.%m.%d")
        story.append(Spacer(1, 60*mm))
        story.append(Paragraph(f"DATE: {d}", self.styles['TitleSmall']))
        
        story.append(PageBreak())
        
        # Pre-process text to remove commas between Evidence tags
        # Logic: Replace "[[Evidence X]],[[Evidence Y]]" with "[[Evidence X]] [[Evidence Y]]" (space instead of comma)
        # or just handle it in the parser.
        # Better: Global regex replace before parsing.
        report_text = re.sub(r'(\]\])\s*[,、]\s*(\[\[Evidence)', r'\1 \2', report_text)
        
        # Content
        story.extend(self.parse_markdown(report_text, snapshots))
        
        try:
            doc.build(story, onFirstPage=self.draw_cover, onLaterPages=self.draw_footer)
            buffer.seek(0)
            return buffer.getvalue(), None
        except Exception as e: return None, str(e)

    def draw_footer(self, canvas, doc):
        canvas.saveState()
        canvas.setFont(self.font_name, 9)
        canvas.setFillColor(colors.gray)
        canvas.drawRightString(190*mm, 15*mm, f"{doc.page}")
        
        # Accent Bar at bottom
        canvas.setFillColor(colors.HexColor('#81D4FA'))
        canvas.rect(0, 0, 10*mm, 297*mm, fill=1, stroke=0) # Ice Blue Side Strip
        
        canvas.restoreState()
    
# Singleton
def generate_pdf(report_text, snapshots, mission_objective="", subtitle=""):
    generator = ReportGenerator()
    return generator.create_pdf(report_text, snapshots, mission_objective, subtitle)
