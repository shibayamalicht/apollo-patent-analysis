import io
import os
import re
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
        self.font_name = "Helvetica" # Default fallback
        self.setup_fonts()
        self.setup_styles()


    def setup_fonts(self):
        if not HAS_REPORTLAB: return
        
        # --- Universal Font Downloader (IPAex Gothic) ---
        # Ensures consistent "TeX-like" quality on Mac/Windows/Cloud
        font_url = "https://moji.or.jp/wp-content/ipafont/IPAexfont/ipaexg00401.ttf" # Example stability, or better use GitHub mirror
        # Actually, let's use a highly stable GitHub raw link or standard path.
        # Ideally we should bundle it, but user asked for robust handling.
        # We will check for 'ipaexg.ttf' in current dir or .fonts dir.
        
        target_font_path = "ipaexg.ttf"
        font_name = "IPAexGothic"
        
        # 1. Check if font exists locally
        if not os.path.exists(target_font_path):
            # Only attempt download if on Linux/Cloud or if system fonts are likely missing
            import platform
            if platform.system() != "Darwin": # Skip on Mac if not needed, but for "Universal" consistency let's try.

                try:
                    import requests
                    url = "https://github.com/googlefonts/ipaex-font/raw/main/ipaexg.ttf" # Stable Google Fonts mirror (unofficial but reliable) OR
                    # Alternative: We can use a public CDN. For now, let's use a very generic placeholder logic or skip if offline.
                    # GitHub direct raw link is usually okay for small assets.
                    # Better link: https://moji.or.jp/wp-content/ipafont/IPAexfont/ipaexg00401.ttf (Official) - might be zipped.
                    
                    # Using a direct raw link to a TTF is safest.
                    url = "https://raw.githubusercontent.com/minoryorg/ipaex-font/master/ipaexg.ttf" # Example mirror
                    
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        with open(target_font_path, 'wb') as f:
                            f.write(response.content)
                        st.success("Font downloaded successfully.")
                except Exception as e:
                    pass

        # Robust Logic: Register if file exists
        if os.path.exists(target_font_path):
            try:
                pdfmetrics.registerFont(TTFont('IPAexGothic', target_font_path))
                self.font_name = 'IPAexGothic'
                return # Success
            except: pass

        # Robust Logic: Try standard System Fonts first, then CID
        self.font_name = "HeiseiKakuGo-W5" # Default CID (Safe on all platforms)
        
        # Try to register a better font if available
        # IPAexGothic or NotoSansCJK are best for TeX look.
        better_fonts = [
            "ipaexg.ttf", # Local
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", # Linux/Cloud
            "/System/Library/Fonts/Hiragino Sans W3.ttc", # Mac
            "C:/Windows/Fonts/meiryo.ttc" # Windows
        ]
        
        for f in better_fonts:
            if os.path.exists(f):
                try:
                    pdfmetrics.registerFont(TTFont('CustomJP', f, subfontIndex=0))
                    self.font_name = 'CustomJP'
                    break
                except: continue
        
        # If still default, use CID
        if self.font_name == "HeiseiKakuGo-W5":
            from reportlab.pdfbase.cidfonts import UnicodeCIDFont
            pdfmetrics.registerFont(UnicodeCIDFont('HeiseiKakuGo-W5'))

    def setup_styles(self):
        if not HAS_REPORTLAB: return
        
        # TeX-Like Clean Typography
        from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
        
        # Base Style
        self.styles = getSampleStyleSheet()
        for style_name in self.styles.byName:
            style = self.styles[style_name]
            style.fontName = self.font_name
        
        # H1: Section (Navy Block)
        self.styles.add(ParagraphStyle(
            name='ReportHeading1',
            parent=self.styles['Heading1'],
            fontName=self.font_name,
            fontSize=16,
            leading=24, # Airy
            spaceBefore=24,
            spaceAfter=12,
            textColor=colors.white,
            backColor=colors.HexColor('#003366'),
            borderPadding=(8, 4, 8, 4), # TB LR
            borderRadius=2,
            keepWithNext=True,
            alignment=TA_LEFT
        ))

        # H2: Subsection (Minimalist Navy)
        self.styles.add(ParagraphStyle(
            name='ReportHeading2',
            parent=self.styles['Heading2'],
            fontName=self.font_name,
            fontSize=13,
            leading=18,
            spaceBefore=18,
            spaceAfter=8,
            textColor=colors.HexColor('#003366'),
            keepWithNext=True
        ))

        # H3: Sub-subsection (Simple Bold Blue)
        self.styles.add(ParagraphStyle(
            name='ReportHeading3',
            parent=self.styles['Heading3'],
            fontName=self.font_name,
            fontSize=11,
            leading=14,
            spaceBefore=12,
            spaceAfter=6,
            textColor=colors.HexColor('#003366'), # Matches H2
            keepWithNext=True
        ))

        # Normal: Justified, High Leading (TeX Style)
        self.styles.add(ParagraphStyle(
            name='ReportNormal',
            parent=self.styles['Normal'],
            fontName=self.font_name,
            fontSize=10.5, # Slightly larger for readability
            leading=17,    # 1.6x leading
            spaceAfter=8,
            alignment=TA_JUSTIFY, # Justified text
            firstLineIndent=0,
        ))
        
        # Bullets
        self.styles.add(ParagraphStyle(
            name='ReportBullet',
            parent=self.styles['ReportNormal'],
            leftIndent=15,
            firstLineIndent=0,
            bulletIndent=5,
            alignment=TA_LEFT # Bullets weird with justify
        ))
        
        # Caption
        self.styles.add(ParagraphStyle(
            name='Caption',
            parent=self.styles['Normal'],
            fontName=self.font_name,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.gray,
            spaceBefore=6,
            spaceAfter=18
        ))
        
        # Title Page Styles
        self.styles.add(ParagraphStyle(
            name='TitlePageTitle',
            fontName=self.font_name,
            fontSize=32,
            leading=42,
            textColor=colors.HexColor('#2c3e50'),
            alignment=TA_LEFT,
        ))
        
        self.styles.add(ParagraphStyle(
            name='TitlePageSubtitle',
            fontName=self.font_name,
            fontSize=14,
            leading=24,
            textColor=colors.HexColor('#7f8c8d'),
            alignment=TA_LEFT,
        ))

    def parse_markdown(self, text, snapshots):
        """Parse with TeX-like handling"""
        flowables = []
        
        def replace_bold(match): return f"<b>{match.group(1)}</b>"
        def replace_italic(match): return f"<i>{match.group(1)}</i>"

        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                flowables.append(Spacer(1, 4*mm))
                continue
            
            # --- Horizontal Rule ---
            # Treat '---', '***', '___' as section breaks
            if re.match(r'^[-*_]{3,}$', line):
                 flowables.append(Spacer(1, 4*mm))
                 # Optional: Draw a line
                 from reportlab.platypus import Flowable
                 class HR(Flowable):
                     def __init__(self, width=100*mm):
                         Flowable.__init__(self)
                         self.width = width
                         self.height = 1*mm
                     def draw(self):
                         self.canv.setStrokeColor(colors.lightgrey)
                         self.canv.line(0, 0, self.width, 0)
                 
                 flowables.append(HR(width=160*mm))
                 flowables.append(Spacer(1, 4*mm))
                 continue
            
            # Image Injection (End of Paragraph Rule)
            # Use capturing group to keep delimiters
            parts = re.split(r'(\[\[Evidence \d+\]\])', line, flags=re.IGNORECASE)
            
            if len(parts) > 1:
                # Iterate parts to handle text and images
                for i, part in enumerate(parts):
                    if not part.strip(): continue
                    
                    ev_match = re.search(r'\[\[Evidence (\d+)\]\]', part, re.IGNORECASE)
                    
                    if ev_match:
                         # --- IMAGE ---
                         ev_id = int(ev_match.group(1)) - 1
                         if 0 <= ev_id < len(snapshots):
                            snap = snapshots[ev_id]
                            if snap.get('image'):
                                try:
                                    img_bytes = snap['image']
                                    rl_img = RLImage(io.BytesIO(img_bytes))
                                    
                                    # Aspect Ratio Fit
                                    img = Image.open(io.BytesIO(img_bytes))
                                    w, h = img.size
                                    aspect = h / float(w)
                                    # TeX style wide images
                                    disp_w = 160*mm
                                    disp_h = disp_w * aspect
                                    if disp_h > 120*mm: 
                                        disp_h = 120*mm; disp_w = disp_h / aspect
                                    
                                    rl_img.drawHeight = disp_h
                                    rl_img.drawWidth = disp_w
                                    
                                    flowables.append(Spacer(1, 5*mm))
                                    flowables.append(rl_img)
                                    flowables.append(Paragraph(f"<b>Fig.{ev_id+1}</b>: {snap['title']}", self.styles['Caption']))
                                    flowables.append(Spacer(1, 5*mm))
                                except: pass
                    else:
                        # --- TEXT ---
                        # Cleanup commas around images
                        # If next part is image (i+1), strip trailing comma
                        if i + 1 < len(parts) and re.match(r'\[\[Evidence \d+\]\]', parts[i+1], re.IGNORECASE):
                             part = part.rstrip('、, \t')
                        
                        # If prev part was image (i-1), strip leading comma
                        if i - 1 >= 0 and re.match(r'\[\[Evidence \d+\]\]', parts[i-1], re.IGNORECASE):
                             part = part.lstrip('、, \t')
                             
                        if not part.strip(): continue

                        proc = re.sub(r'\*\*(.*?)\*\*', replace_bold, part)
                        flowables.append(Paragraph(proc, self.styles['ReportNormal']))
                continue

            # Standard Text
            proc_line = re.sub(r'\*\*(.*?)\*\*', replace_bold, line)
            
            style = self.styles['ReportNormal']
            
            if line.startswith('# '):
                 if flowables: flowables.append(PageBreak())
                 style = self.styles['ReportHeading1']
                 proc_line = proc_line[2:].strip()
            elif line.startswith('## '):

                 if flowables: flowables.append(PageBreak())
                 style = self.styles['ReportHeading2']
                 proc_line = proc_line[3:].strip()
            elif line.startswith('### '):
                 # H3: Use the blue-colored style
                 style = self.styles['ReportHeading3']
                 proc_line = proc_line[4:].strip()
            elif line.startswith('#### '):
                 style = self.styles['ReportNormal']
                 proc_line = f"<b>{proc_line[5:].strip()}</b>"
            elif line.startswith('- ') or line.startswith('* '):
                 style = self.styles['ReportBullet']
                 proc_line = '• ' + proc_line[2:].strip()
            
            flowables.append(Paragraph(proc_line, style))
            
        return flowables

    def draw_cover(self, canvas, doc):
        """TeX/Swiss Style Cover"""
        canvas.saveState()
        
        # 1. Geometry (Left Stripe)
        canvas.setFillColor(colors.HexColor('#003366'))
        canvas.rect(0, 0, 15*mm, 297*mm, fill=1, stroke=0)
        
        # 2. Top Line
        canvas.setLineWidth(2)
        canvas.setStrokeColor(colors.HexColor('#003366'))
        canvas.line(30*mm, 240*mm, 190*mm, 240*mm)
        
        # 3. Footer
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(colors.gray)
        canvas.drawCentredString(115*mm, 20*mm, "CONFIDENTIAL / INTERNAL USE ONLY")
        
        canvas.restoreState()

    def draw_footer(self, canvas, doc):
        """Draw footer on subsequent pages"""
        canvas.saveState()
        # Page Number
        canvas.setFont(self.font_name, 9)
        canvas.setFillColor(colors.gray)
        canvas.drawRightString(190*mm, 15*mm, f"Page {doc.page}")
        
        # Header Line
        navy = colors.HexColor('#003366')
        canvas.setFillColor(navy)
        canvas.rect(20*mm, 280*mm, 170*mm, 1*mm, fill=1, stroke=0)
        
        canvas.restoreState()

    def create_pdf(self, report_text, snapshots, mission_objective=""):
        if not HAS_REPORTLAB: return None, "No ReportLab"
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            rightMargin=25*mm, leftMargin=35*mm, # TeX optimal margins
            topMargin=25*mm, bottomMargin=25*mm
        )
        
        story = []
        
        # Cover Content
        story.append(Spacer(1, 70*mm))
        story.append(Paragraph("VOYAGER STRATEGIC REPORT", self.styles['TitlePageSubtitle']))
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("Market Intelligence & IP Analysis", self.styles['TitlePageTitle']))
        
        import datetime
        d = datetime.datetime.now().strftime("%B %d, %Y")
        story.append(Spacer(1, 10*mm))
        story.append(Paragraph(f"Generated on {d}", self.styles['TitlePageSubtitle']))
        
        story.append(PageBreak())
        story.extend(self.parse_markdown(report_text, snapshots))
        
        try:
            doc.build(story, onFirstPage=self.draw_cover, onLaterPages=self.draw_footer)
            buffer.seek(0)
            return buffer.getvalue(), None
        except Exception as e: return None, str(e)

# Singleton helper
def generate_pdf(report_text, snapshots, mission_objective=""):
    generator = ReportGenerator()
    return generator.create_pdf(report_text, snapshots, mission_objective)
