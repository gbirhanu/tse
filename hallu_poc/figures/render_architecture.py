"""Create an editable scientific architecture figure. Requires reportlab.

Run: python render_architecture.py --out-dir .
The PNG preview can be rendered from the PDF using pdftoppm.
"""
import argparse
import math
from pathlib import Path

from reportlab.graphics import renderPDF, renderSVG
from reportlab.graphics.shapes import Drawing, Line, Polygon, PolyLine, Rect, String
from reportlab.lib.colors import HexColor, Color
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


W, H = 1800, 1100
INK = '#172C3C'
MUTED = '#516571'
LINE = '#CCD6DC'
TEAL = '#176C78'
TEAL_LIGHT = '#EFF7F8'
RUST = '#AC6335'
RUST_LIGHT = '#FBF4EE'
PAPER = '#FFFFFF'
GRAY = '#F5F7F9'
FONT = 'Helvetica'
BOLD = 'Helvetica-Bold'
ITALIC = 'Helvetica-Oblique'

font_dir = Path('C:/Windows/Fonts')
if (font_dir / 'arial.ttf').exists():
    for name, filename in [('Arial', 'arial.ttf'), ('Arial-Bold', 'arialbd.ttf'), ('Arial-Italic', 'ariali.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    FONT, BOLD, ITALIC = 'Arial', 'Arial-Bold', 'Arial-Italic'

d = Drawing(W, H)

def box(x, y, w, h, fill=PAPER, stroke=LINE, radius=6, width=1.2, dashed=False):
    r = Rect(x, H-y-h, w, h, rx=radius, ry=radius,
             fillColor=HexColor(fill) if fill else None,
             strokeColor=HexColor(stroke) if stroke else None, strokeWidth=width)
    if dashed:
        r.strokeDashArray = [5, 4]
    d.add(r)

def text(x, y, value, size=18, color=INK, bold=False, align='start', italic=False):
    # y is a top-down baseline. All labels remain editable SVG text.
    d.add(String(x, H-y, value, fontName=ITALIC if italic else BOLD if bold else FONT,
                 fontSize=size, fillColor=HexColor(color), textAnchor=align))

def lines(x, y, values, size=18, step=25, **kwargs):
    for j, value in enumerate(values):
        text(x, y+j*step, value, size=size, **kwargs)

def equation(x, y, runs, size=20):
    """Typeset subscripts without relying on unusual Unicode font glyphs."""
    for value, scale, offset in runs:
        text(x, y+offset, value, size=size*scale)
        x += pdfmetrics.stringWidth(value, FONT, size*scale)
    return x

def path(points, color=MUTED, width=1.6, dashed=False, arrow=False):
    p = PolyLine([z for x, y in points for z in (x, H-y)], strokeColor=HexColor(color),
                 strokeWidth=width, fillColor=None)
    if dashed:
        p.strokeDashArray = [5, 4]
    d.add(p)
    if arrow:
        (x1, y1), (x2, y2) = points[-2:]
        angle = math.atan2(y2-y1, x2-x1)
        length, half = 9, 4
        bx, by = x2-length*math.cos(angle), y2-length*math.sin(angle)
        coords = [(x2, y2), (bx+half*math.sin(angle), by-half*math.cos(angle)),
                  (bx-half*math.sin(angle), by+half*math.cos(angle))]
        d.add(Polygon([z for x, y in coords for z in (x, H-y)],
                      fillColor=HexColor(color), strokeColor=None))

def strip(x, y, values, color, cell=17, gap=4, height=12):
    c = HexColor(color)
    for i, v in enumerate(values):
        fill = Color(1-(1-c.red)*v, 1-(1-c.green)*v, 1-(1-c.blue)*v)
        d.add(Rect(x+i*(cell+gap), H-y-height, cell, height, rx=1, ry=1,
                   fillColor=fill, strokeColor=HexColor(LINE), strokeWidth=.5))

def target_plot(x, y, values, color, title):
    text(x, y, title, size=18, bold=True, color=color)
    top, bottom, width = y+21, y+115, 315
    path([(x, top), (x, bottom), (x+width, bottom)], color=LINE, width=1)
    text(x-9, top+5, '1', size=13, color=MUTED, align='end')
    text(x-9, bottom+4, '0', size=13, color=MUTED, align='end')
    for i, value in enumerate(values):
        xx = x+14+i*32
        hh = max(1.8, value*85)
        box(xx, bottom-hh, 22, hh, fill=color, stroke=None, radius=1)
    text(x+width/2, bottom+24, 'Answer-token position i', size=14, color=MUTED, align='middle')

# Page and title.
box(0, 0, W, H, fill=PAPER, stroke=None, radius=0)
text(55, 59, 'Lightweight hallucination span detection', size=34, bold=True)
text(55, 91, 'Source-conditioned token classification with exact character-span output', size=19, color=MUTED)
path([(55, 111), (1745, 111)], color=LINE, width=1)

text(55, 148, 'a', size=24, bold=True)
text(85, 148, 'Inference architecture', size=23, bold=True)
text(1745, 148, 'Ensemble configurations are candidates for evaluation', size=16, color=MUTED,
     align='end', italic=True)

# Left-to-right data flow: prompt and answer, aligned tokens, independent encoders.
box(55, 187, 215, 109)
text(72, 215, 'SOURCE PROMPT', size=13, bold=True, color=MUTED)
lines(72, 246, ['The trial enrolled', '120 adults.'], size=19, step=25)
box(55, 322, 215, 109)
text(72, 350, 'LLM ANSWER', size=13, bold=True, color=MUTED)
lines(72, 381, ['The trial enrolled', '150 adults.'], size=19, step=25)
text(55, 458, 'Illustrative input', size=14, color=MUTED, italic=True)
path([(270, 242), (293, 242), (293, 310)], color=MUTED)
path([(270, 377), (293, 377), (293, 310), (320, 310)], arrow=True)

box(320, 231, 175, 158, fill=GRAY)
text(407.5, 263, 'Pair tokenizer', size=19, bold=True, align='middle')
lines(407.5, 294, ['Prompt + answer', '4,096-token context'], size=16, align='middle', step=23)
path([(338, 331), (477, 331)], color=LINE, width=1)
lines(407.5, 351, ['Answer offsets', '[start, end)'], size=16, align='middle', step=23)
text(407.5, 417, 'Truncate prompt only', size=14, color=MUTED, align='middle')

text(545, 185, 'INDEPENDENT MODEL WEIGHTS', size=13, color=MUTED, bold=True)
for y, color, fill, title, values in [
        (202, TEAL, TEAL_LIGHT, 'Hard-target models', [.1,.08,.18,.85,.94,.75,.08,.05]),
        (326, RUST, RUST_LIGHT, 'Soft-target models', [.08,.12,.3,.76,.9,.69,.17,.08])]:
    box(552, y-7, 303, 103, fill=fill, stroke=color, width=.8)
    box(548, y-3, 303, 103, fill=fill, stroke=color, width=.8)
    box(545, y, 303, 103, fill=fill, stroke=color, width=1.4)
    text(561, y+27, title, size=20, color=color, bold=True)
    text(561, y+52, 'ModernBERT-base · 0.15B each', size=16)
    text(561, y+75, 'Seeds 42 / 43 / 44', size=15, color=MUTED)
    strip(653, y+84, values, color, cell=15, gap=4, height=8)

box(545, 454, 303, 67, fill=GRAY, stroke=MUTED, dashed=True)
text(561, 480, 'Published LettuceDetect-large', size=17, bold=True)
text(561, 504, 'Optional member · 0.4B', size=16, color=MUTED)

path([(495, 310), (518, 310)], color=MUTED)
path([(518, 253), (518, 378)], color=MUTED)
path([(518, 253), (545, 253)], color=TEAL, arrow=True)
path([(518, 378), (545, 378)], color=RUST, arrow=True)
path([(518, 378), (518, 488), (545, 488)], color=MUTED, dashed=True, arrow=True)

path([(848, 253), (890, 253), (890, 319)], color=TEAL)
path([(848, 378), (890, 378), (890, 319), (928, 319)], color=RUST, arrow=True)
path([(848, 488), (890, 488), (890, 378)], color=MUTED, dashed=True)

box(928, 242, 215, 174, fill=TEAL_LIGHT, stroke=TEAL, width=1.4)
text(1035.5, 271, 'Token probabilities', size=19, bold=True, align='middle')
text(1035.5, 299, 'Two-class softmax', size=16, color=MUTED, align='middle')
path([(944, 317), (1127, 317)], color=LINE, width=1)
text(1035.5, 343, 'Equal-weight mean', size=19, bold=True, align='middle', color=TEAL)
equation(945, 371, [('p',1,0), ('i',.58,4), (' = (1/M) ',1,0), ('Σ',1.1,0),
                     ('m',.58,4), (' p',1,0), ('i',.58,4), ('(m)',.58,-8)])
path([(945, 354), (956, 354)], color=INK, width=.9)
strip(955, 387, [.08,.1,.24,.81,.92,.72,.13,.07], TEAL, cell=16, gap=4, height=10)
lines(1035.5, 447, ['Same tokenizer and', 'identical ordered offsets'], size=15, color=MUTED, align='middle', step=22)
path([(1143, 319), (1195, 319)], arrow=True)

box(1195, 242, 225, 174)
text(1307.5, 272, 'Span decoder', size=21, bold=True, align='middle')
lines(1213, 308, ['1   Probability threshold', '2   Merge token gaps', '3   Optional word snap'], size=16.5, step=32)
text(1307.5, 447, 'Per-task calibration', size=15, color=MUTED, align='middle')
text(1307.5, 470, 'See protocol in panel c', size=15, color=MUTED, align='middle')
path([(1420, 319), (1475, 319)], arrow=True)

box(1475, 218, 270, 222)
text(1494, 250, 'CHARACTER-SPAN OUTPUT', size=13, color=MUTED, bold=True)
text(1494, 289, 'The trial enrolled', size=21)
box(1490, 303, 47, 29, fill='#F8DFCD', stroke=None, radius=2)
text(1494, 325, '150', size=21, bold=True, color='#8D4729')
text(1542, 325, 'adults.', size=21)
path([(1494, 349), (1726, 349)], color=LINE, width=1)
text(1494, 379, '[19, 22)  →  “150”', size=21, bold=True)
text(1494, 410, 'Unsupported answer characters', size=15, color=MUTED)
text(1475, 470, 'Zero-based, end-exclusive offsets', size=14, color=MUTED)

path([(55, 559), (1745, 559)], color=LINE, width=1)

# Training supervision, intentionally separate from inference arrows.
text(55, 604, 'b', size=24, bold=True)
text(85, 604, 'Training supervision', size=23, bold=True)
text(55, 635, 'Annotated character spans supervise answer tokens only.', size=17, color=MUTED)
target_plot(82, 675, [0,0,0,1,1,1,0,0,0], TEAL, 'Hard target  yᵢ = hᵢ')
target_plot(491, 675, [.03,.06,.12,.87,.95,.87,.12,.06,.03], RUST, 'Soft target  yᵢ = 0.6hᵢ + 0.4qᵢ')
text(55, 847, 'hᵢ : overlap with a gold span', size=17)
text(55, 875, 'qᵢ : boundary-shift coverage weighted by J + J²', size=17)
text(55, 900, 'J is word-level Jaccard. Target profiles above are schematic.', size=15, color=MUTED)
box(55, 926, 810, 66, fill=GRAY)
text(73, 952, 'ANSWER-ONLY CROSS-ENTROPY   (A: answer tokens)', size=12, bold=True, color=MUTED)
equation(73, 978, [('L = − mean',1,0), ('i in A',.55,5),
                    (' [ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ) ]',1,0)], size=21)

path([(895, 589), (895, 992)], color=LINE, width=1)

# Held-out protocol. No arrow returns from test evaluation to calibration.
text(928, 604, 'c', size=24, bold=True)
text(958, 604, 'Calibration and evaluation', size=23, bold=True)
text(928, 635, 'Our models: split RAGTruth training data by source.', size=17, color=MUTED)
for x, title, sub in [(928, '90% train sources', ['Fit independent', 'classifier weights']),
                       (1208, '10% dev sources', ['Select task-specific', 'decoder settings']),
                       (1488, 'RAGTruth test', ['Evaluate frozen pipeline', '2,700 answers'])]:
    box(x, 662, 257, 105, fill=GRAY)
    text(x+16, 692, title, size=19, bold=True)
    lines(x+16, 721, sub, size=16, color=MUTED, step=23)
path([(1185, 714), (1208, 714)], arrow=True)
path([(1465, 714), (1488, 714)], arrow=True)
text(928, 800, 'LettuceDetect exception', size=17, bold=True)
lines(928, 826, ['Any ensemble including published LD uses fixed decoding:',
                'threshold 0.5, gap 0, no word snapping.'], size=16, step=23)
text(928, 880, 'Two-fold cross-fitting is restricted to published LD alone.', size=15, color=MUTED)
box(928, 905, 817, 87, fill=TEAL_LIGHT, stroke=None)
text(946, 932, 'Primary metric: LettuceDetect-compatible character span F1', size=17, bold=True)
text(946, 958, 'Paired bootstrap over test sources · 95% confidence interval', size=16)
text(946, 982, 'Retain a change only when its gain interval is entirely above zero.', size=16)

path([(55, 1022), (1745, 1022)], color=LINE, width=1)
text(55, 1052, 'Architecture of the current pipeline and Part C ensemble experiment. Parameter counts are per model.', size=15, color=MUTED)
text(55, 1078, 'Soft targets retain the legacy boundary-shift implementation. Ensemble gains await full evaluation.', size=15, color=MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', type=Path, default=Path('.'))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = args.out_dir / 'hallu_architecture_v1'
    renderPDF.drawToFile(d, str(base.with_suffix('.pdf')), title='Lightweight hallucination span detection',
                         author='Hallucination span detection research project')
    renderSVG.drawToFile(d, str(base.with_suffix('.svg')))
    caption = (
        'Figure. Lightweight hallucination span detection. (a) A source prompt and an LLM answer '
        'are pair-tokenized; the prompt is truncated as needed to fit the 4,096-token context. '
        'Independent hard- and soft-target ModernBERT-base classifiers produce two-class softmax '
        'probabilities for answer tokens. Predeclared ensembles average probabilities only when '
        'tokenizers and ordered character offsets agree. The decoder thresholds probabilities, '
        'merges gaps measured in tokens, and optionally snaps boundaries to words, returning '
        'zero-based, end-exclusive character intervals. The displayed answer is illustrative. '
        '(b) Both target variants use answer-only cross-entropy. Hard labels mark token overlap '
        'with gold character spans; soft labels mix 0.6 hard labels with 0.4 boundary-shift coverage '
        'weighted by word-level Jaccard J + J squared. Target profiles are schematic. The current '
        'legacy implementation is not limited to symmetric five-word boundary shifts. '
        '(c) Our models use a 10% source-held-out dev split for task-specific decoder tuning. '
        'Because published LettuceDetect was trained on all RAGTruth training sources, any '
        'ensemble containing it uses fixed baseline decoding; two-fold cross-fitting is restricted '
        'to published LettuceDetect alone. Evaluation uses the LettuceDetect-compatible character '
        'span F1 and paired 95% bootstrap intervals over test sources. Seeds 43 and 44 and the '
        'ensemble configurations are planned or running candidates, not evidence of an accepted '
        'improvement. Parameter counts are per ensemble member. The optional character stacker '
        'and future span-F1 loss, 8,192-token training, and PsiloQA pretraining are outside this figure.\n'
    )
    base.with_suffix('.caption.txt').write_text(caption, encoding='utf-8')
    print('Created', base.with_suffix('.pdf'))
    print('Created', base.with_suffix('.svg'))


if __name__ == '__main__':
    main()
