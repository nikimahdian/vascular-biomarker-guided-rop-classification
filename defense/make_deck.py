"""ROP defense deck - v3.

Changes over v2:
  * speaker notes rewritten from the defence-course material (files 14, 15, 17 of the course):
    longer, defence-oriented, leading with contribution and evidence;
    limitations are stated as SCOPE BOUNDARIES with a defence pivot, never as self-criticism.
  * bullets stay short titles; Latin tokens keep their own Times New Roman runs.
"""
import re
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

FIG = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL\thesis_v9\figures\report")
OUT = Path(r"C:\Users\nikim\OneDrive\Desktop\ROP_Defense_Presentation_v4.pptx")

NAVY = RGBColor(0x1B, 0x2A, 0x41)
GREY = RGBColor(0x50, 0x50, 0x50)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
FA, EN = "B Nazanin", "Times New Roman"
LATIN = re.compile(r"([A-Za-z][A-Za-z0-9\-\._/]*)")

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def rtl(p, align=PP_ALIGN.RIGHT):
    p._p.get_or_add_pPr().set("rtl", "1")
    p.alignment = align


def write(p, text, size, bold=False, color=NAVY):
    p._p.get_or_add_pPr().set("rtl", "1")
    pos = 0
    for m in LATIN.finditer(text):
        if m.start() > pos:
            r = p.add_run(); r.text = text[pos:m.start()]
            r.font.name = FA; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
        r = p.add_run(); r.text = m.group(1)
        r.font.name = EN; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
        pos = m.end()
    if pos < len(text):
        r = p.add_run(); r.text = text[pos:]
        r.font.name = FA; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color


def textbox(slide, l, t, w, h):
    tb = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    tb.text_frame.word_wrap = True
    return tb.text_frame


def title_bar(slide, title, sub=None):
    bar = slide.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width, Inches(1.12))
    bar.fill.solid(); bar.fill.fore_color.rgb = NAVY; bar.line.fill.background()
    bar.shadow.inherit = False
    tf = bar.text_frame
    tf.margin_right = Inches(0.5); tf.margin_left = Inches(0.5)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    write(tf.paragraphs[0], title, 30, bold=True, color=WHITE)
    if sub:
        write(textbox(slide, 0.7, 1.22, 12.0, 0.5).paragraphs[0], sub, 17, color=GREY)


def set_notes(slide, text):
    tf = slide.notes_slide.notes_text_frame
    tf.text = text
    for p in tf.paragraphs:
        rtl(p)


def bullet_slide(title, bullets, notes, sub=None):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, sub)
    tf = textbox(s, 0.8, 1.95, 11.8, 4.9)
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        write(p, "▪ " + b, 24)
        p.space_after = Pt(16)
    set_notes(s, notes)
    return s


def figure_slide(title, img, bullets, notes, sub=None):
    s = prs.slides.add_slide(BLANK)
    title_bar(s, title, sub)
    s.shapes.add_picture(str(FIG / img), Inches(0.55), Inches(1.85), width=Inches(6.6))
    tf = textbox(s, 7.35, 1.95, 5.45, 4.9)
    for i, b in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        write(p, "▪ " + b, 21)
        p.space_after = Pt(16)
    set_notes(s, notes)
    return s


# ---------------------------------------------------------------- notes (course-sourced)
import sys as _sys
_sys.path.insert(0, r"C:\Users\nikim\OneDrive\Desktop\ROP_FINAL")
from _notes_v4 import build as _build_notes  # noqa: E402

N = _build_notes()


def set_notes(slide, text):
    tf = slide.notes_slide.notes_text_frame
    tf.clear()
    first = True
    for line in str(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        write(p, line, 14, color=RGBColor(0x22, 0x22, 0x22))
        p.space_after = Pt(5)


# ================================================================ slides
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, Inches(0), Inches(0), prs.slide_width, prs.slide_height)
bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background(); bg.shadow.inherit = False
tf = textbox(s, 1.0, 2.0, 11.3, 1.4)
write(tf.paragraphs[0], "تشخیص بیماری پلاس در رتینوپاتی نارس", 38, bold=True, color=WHITE)
tf.paragraphs[0].alignment = PP_ALIGN.CENTER
p = tf.add_paragraph()
write(p, "ارزیابی سهم بازنمایی عروقی فضایی و نشانگرهای اسکالر عروقی در کنار بازنمایی تصویر", 19,
      color=RGBColor(0xD5, 0xDD, 0xE8))
p.alignment = PP_ALIGN.CENTER
tf = textbox(s, 1.0, 4.5, 11.3, 1.6)
for i, (line, sz, bold) in enumerate([("نیکی مهدیان", 22, True),
                                      ("استاد راهنما: دکتر تقی راد", 17, False),
                                      ("دانشکده‌ی مهندسی برق - دانشگاه صنعتی خواجه نصیرالدین طوسی", 16, False),
                                      ("پروژه‌ی کارشناسی", 16, False)]):
    pp = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
    write(pp, line, sz, bold=bold, color=WHITE if bold else RGBColor(0xB9, 0xC4, 0xD4))
    pp.alignment = PP_ALIGN.CENTER
set_notes(s, N[1])

bullet_slide("مسئله‌ی بالینی", ["رتینوپاتی نارس و ROP", "بیماری Plus", "مرز Pre-Plus",
                              "خروجی مدل: احتمال سه‌رده‌ای"], N[2])
bullet_slide("پرسش پژوهش", ["چهار پرسش پژوهش", "یک پروتکل واحد", "آزمون قفل‌شده"], N[3])
figure_slide("جمعیت، منابع و تقسیم", "main/D1_source_class_distribution.png",
             ["۸۸۶۲ مورد، ۴۱۴ گروه", "سه مرکز تصویربرداری", "تقسیم ۶۲۰۳ / ۱۳۲۸ / ۱۳۳۱",
              "عدم‌توازن منبع"], N[4], sub="منبع بزرگ، عدد تجمیعی را جهت می‌دهد")
bullet_slide("ممیزی داده و جلوگیری از نشت",
             ["استقلال در سطح گروه", "تعریف گروه در Farabi", "نبود هم‌پوشانی", "قفل اثر انگشت داده"], N[5])
figure_slide("زنجیره‌ی عروقی و پنج نشانگر", "methods/M2_biomarker_pipeline.png",
             ["پنج نشانگر نهایی", "مخرج مساحت محتوا", "خط لوله‌ی تثبیت‌شده"],
             N[6], sub="پنج خلاصه‌ی عددی با معنای هندسی روشن")
figure_slide("طراحی مدل‌ها", "methods/M3_model_architectures.png",
             ["هفت ورودی اصلی", "XGBoost با تنظیم ثابت"], N[7],
             sub="یک پیکربندی ثابت برای همه‌ی مدل‌های برداری")
bullet_slide("روش ارزیابی", ["سه رژیم جدا", "Bootstrap زوجی طبقه‌محور", "پنج معیار", "قفل آزمون"],
             N[8], sub="سه رژیم جدا که هرگز با هم جمع نمی‌شوند")

s = prs.slides.add_slide(BLANK)
title_bar(s, "نتایج آزمون کانونی", "آزمون کانونی با ۱۳۳۱ تصویر")
headers = ["مدل", "AUC", "دقت متعادل", "F1 ماکرو", "Brier", "ECE"]
rows = [["نشانگرهای اسکالر", "۰٫۶۵۹۶", "۰٫۴۹۸۵", "۰٫۴۳۴۶", "۰٫۵۶۷۵", "۰٫۰۵۰۲"],
        ["شبکه‌ی تصویری", "۰٫۹۱۳۳", "۰٫۶۹۸۶", "۰٫۶۸۴۸", "۰٫۲۶۲۳", "۰٫۰۴۹۷"],
        ["بازنمایی تصویری", "۰٫۹۲۴۹", "۰٫۷۲۳۴", "۰٫۷۰۸۷", "۰٫۲۷۶۳", "۰٫۱۰۲۸"],
        ["تصویر + نشانگرها", "۰٫۹۲۶۰", "۰٫۷۲۰۴", "۰٫۷۰۹۳", "۰٫۲۶۹۰", "۰٫۰۹۸۶"],
        ["تصویر + عروق", "۰٫۹۳۲۸", "۰٫۷۳۵۰", "۰٫۷۲۴۶", "۰٫۲۴۸۶", "۰٫۰۹۲۱"],
        ["تصویر + عروق + نشانگرها", "۰٫۹۳۴۹", "۰٫۷۳۸۹", "۰٫۷۲۹۲", "۰٫۲۴۶۵", "۰٫۰۹۲۱"]]
tbl = s.shapes.add_table(7, 6, Inches(0.55), Inches(1.85), Inches(12.2), Inches(3.0)).table
widths = [3.4, 1.5, 1.7, 1.7, 1.4, 1.4]
for i, w in enumerate(widths):
    tbl.columns[i].width = Emu(int(Inches(12.2) * w / sum(widths)))
for j, h in enumerate(headers):
    c = tbl.cell(0, j); c.text = ""
    write(c.text_frame.paragraphs[0], h, 15, bold=True, color=WHITE)
    c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    c.fill.solid(); c.fill.fore_color.rgb = NAVY
for i, row in enumerate(rows, start=1):
    for j, v in enumerate(row):
        c = tbl.cell(i, j); c.text = ""
        write(c.text_frame.paragraphs[0], v, 15)
        c.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
        c.fill.solid(); c.fill.fore_color.rgb = WHITE if i % 2 else LIGHT
write(textbox(s, 0.8, 5.5, 11.7, 0.9).paragraphs[0],
      "AUC، دقت متعادل و F1 ماکرو بزرگ‌تر‌بهتر و Brier و ECE کوچک‌تر‌بهتر هستند. "
      "هیچ امتیاز ترکیبی ساخته نشد.", 15, color=GREY)
set_notes(s, N[9])

figure_slide("توان رتبه‌بندی مدل‌ها", "main/R1_model_auc_comparison.png",
             ["AUC تصویری: ۰٫۹۲۵", "افزودن عروق: ۰٫۹۳۳", "برآورد نقطه‌ای، شاهد نیست"], N[10],
             sub="نشانگر تنها، ضعیف‌تر از هر مدل تصویری است")
figure_slide("سهم افزایشی هر مؤلفه", "main/C1_incremental_auc_forest.png",
             ["نشانگرها: بدون شاهد", "عروق: شاهد مثبت", "نشانگر پس از عروق: بدون شاهد"], N[11],
             sub="نتیجه‌ی مرکزی پایان‌نامه")
figure_slide("پایداری سهم عروقی در معماری‌های متفاوت",
             "main/C3_vessel_complementarity_across_rgb_architectures.png",
             ["تکرار در دو بازنمایی", "برتری معماری ادعا نمی‌شود"], N[12],
             sub="سهم عروقی، تکرارپذیر در دو استخراج‌کننده‌ی متفاوت")
figure_slide("کیفیت احتمال و کالیبراسیون", "main/P2_reliability_main.png",
             ["کالیبراسیون ناقص", "بهبود AUC و Brier", "معیارهای آستانه‌ای جدا"], N[13],
             sub="رتبه‌بندی و احتمال، دو لایه‌ی جدا")
figure_slide("عملکرد به تفکیک کلاس", "main/R3_preplus.png",
             ["Pre-Plus سخت‌ترین کلاس", "Plus دقیق‌ترین کلاس"], N[14],
             sub="Pre-Plus، سخت‌ترین کلاس برای همه‌ی مدل‌ها")
figure_slide("شواهد پنج نشانگر در همه‌ی رژیم‌ها", "main/B1_biomarker_incremental_auc_forest.png",
             ["شش آزمون مستقل", "نتیجه‌ی منفی پایدار"], N[15], sub="شش آزمون، یک نتیجه‌ی سازگار")
figure_slide("آزمایش‌های ثانویه‌ی مدل‌های عصبی", "appendix/N1a_task9_loss.png",
             ["بیش‌برازش مشترک", "توقف مبتنی بر شاهد"], N[16], sub="سه آزمایش ثانویه با یک الگوی مشترک")
figure_slide("تعمیم به منبع دیده‌نشده", "main/S2_vessel_complementarity_heldout.png",
             ["دو فولد مثبت", "فولد Plus محدود", "افت سطح عملکرد"], N[17],
             sub="اطلاعات منبع هدف در هیچ مرحله‌ای استفاده نشد")
figure_slide("جابه‌جایی دامنه", "main/DS1b_shift_vessel.png",
             ["جابه‌جایی اندازه‌گیری‌شده", "بیشترین جابه‌جایی، بیشترین سود"], N[18],
             sub="جابه‌جایی اندازه‌گیری شد، نه فرض")
figure_slide("آزمون سازگارسازی دامنه", "main/DA1a_domain_alignment_threeclass.png",
             ["جداسازی جابه‌جایی از پیامد", "نتیجه محدود به تنظیم ثابت"], N[19],
             sub="جابه‌جایی کم شد؛ عملکرد بیماری نه")
figure_slide("آزمون معماری جایگزین", "main/R5_task13_paired_auc.png",
             ["کنترل ضعیف‌تر", "عروق: مکمل"], N[20], sub="سهم عروقی، مقاوم به تغییر معماری تصویری")
bullet_slide("نوآوری و سهم علمی",
             ["تفکیک نشانگر از ساختار فضایی", "اثبات زوجی مکمل‌بودن عروق", "تکرار بین‌منبعی",
              "پایداری معماری", "حفظ نتیجه‌ی منفی", "ممیزی داده و منشأ"], N[21])
bullet_slide("محدودیت‌ها به‌عنوان مرز ادعا",
             ["مطالعه‌ی گذشته‌نگر", "گروه Farabi در سطح معاینه", "آزمون مرجع در تحلیل‌های ثانویه",
              "نبود مرجع متخصص جداسازی", "دامنه‌ی پنج نشانگر", "افت بین‌منبعی"], N[22])
bullet_slide("نتیجه‌ی نهایی",
             ["عروق فضایی: اطلاعات مکمل", "تکرار در منبع دیده‌نشده",
              "نشانگرها: بدون ارزش افزوده‌ی تکرارپذیر", "بدون ادعای آمادگی بالینی"], N[23])
bullet_slide("پرسش‌های محتمل داور",
             ["چرا عدد تجمیعی نتیجه‌ی اصلی نیست؟", "چرا آموزش کامل ادامه نیافت؟",
              "چرا سازگارسازی دامنه نتیجه نداد؟", "نوآوری در یک جمله چیست؟"], N[24],
             sub="پاسخ کامل در یادداشت همین اسلاید")

prs.save(str(OUT))
print("SLIDES", len(prs.slides._sldIdLst))
print("SAVED", OUT)
