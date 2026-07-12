"""Generate the clinician one-page PDF brief. Run: python docs/make_onepager.py"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

HERE = Path(__file__).resolve().parent
FIG = HERE.parent / "parkigait" / "figures" / "clinical_eval.png"
OUT = HERE / "ParkiGait_onepager.pdf"

ss = getSampleStyleSheet()
H = ParagraphStyle("H", parent=ss["Title"], fontSize=16, spaceAfter=2, textColor=colors.HexColor("#12324f"))
SUB = ParagraphStyle("SUB", parent=ss["Normal"], fontSize=8.5, alignment=TA_CENTER, textColor=colors.HexColor("#7a1020"), spaceAfter=8)
HD = ParagraphStyle("HD", parent=ss["Heading2"], fontSize=10.5, spaceBefore=7, spaceAfter=2, textColor=colors.HexColor("#12324f"))
BODY = ParagraphStyle("BODY", parent=ss["Normal"], fontSize=8.7, leading=11.5)
SMALL = ParagraphStyle("SMALL", parent=ss["Normal"], fontSize=7.3, leading=9, textColor=colors.HexColor("#555"))


def p(t, s=BODY):
    return Paragraph(t, s)


doc = SimpleDocTemplate(str(OUT), pagesize=letter, topMargin=0.5 * inch,
                        bottomMargin=0.4 * inch, leftMargin=0.6 * inch, rightMargin=0.6 * inch)
E = []
E.append(p("ParkiGait", H))
E.append(p("On-device gait screening for Parkinson's &mdash; research prototype, NOT a medical device", SUB))

E.append(p("What it is", HD))
E.append(p("A tool that takes a short walking video, extracts a skeleton on the device, computes "
           "clinically grounded gait features (walking speed, cadence, stride length, joint range of "
           "motion, arm swing, trunk flexion, step to step variability, freezing index), and flags "
           "possible parkinsonian gait for a clinician to review. It runs fully on a phone or laptop "
           "with no cloud, so patient video never leaves the device."))

E.append(p("Results on real clinical data", HD))
E.append(p("Trained and evaluated on the public CARE-PD dataset (110 patients with clinician MDS-UPDRS "
           "gait scores, ~2953 walks). Strict subject level cross validation, 95% confidence intervals "
           "by patient level bootstrap.", SMALL))
data = [
    ["Task", "Result"],
    ["Detect impaired gait (screening) &mdash; AUC", "0.86 to 0.87 (95% CI 0.82 to 0.90), permutation p < 0.005"],
    ["External validation (a hospital never seen in training)", "pooled AUC 0.77 (0.81 to 0.86 at the larger sites)"],
    ["Severity staging (UPDRS-gait 0 to 3)", "quadratic weighted kappa 0.62, 95% within one level"],
    ["Screening operating point", "sensitivity 0.90, specificity 0.59, PPV 0.75, NPV 0.81"],
    ["Test-retest reliability (repeated walks)", "ICC 0.63, calibrated probability (Brier 0.15)"],
    ["Speed / memory / privacy", "~27 ms per frame, under 400 MB, on-device (no cloud)"],
]
t = Table([[p(c[0], SMALL), p(c[1], SMALL)] for c in data], colWidths=[3.0 * inch, 4.2 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#12324f")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, 0), 8),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cfd6dd")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f8")]),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]))
E.append(t)
E.append(p("The strongest single feature is walking speed, followed by stride length and joint range of "
           "motion, the same signs a clinician looks for. Leakage was independently audited and a "
           "permutation test confirms the result is not chance.", SMALL))

if FIG.exists():
    E.append(Spacer(1, 4))
    E.append(Image(str(FIG), width=5.3 * inch, height=5.3 * 10 / 12 * inch))

E.append(p("What it is not", HD))
E.append(p("Not a diagnosis, not a replacement for a clinician or the standard exam, and not ready for "
           "any patient care decision. It is a research and screening aid.", BODY))

E.append(p("What I am seeking", HD))
E.append(p("Feedback from a clinician on where this could genuinely help, and if you are open to it a "
           "small research collaboration under ethics approval to compare the tool's flag to your own "
           "assessment on anonymized gait videos.", BODY))

E.append(Spacer(1, 4))
E.append(p("Parv Mehndiratta &nbsp;|&nbsp; open source code and full honest documentation: "
           "github.com/0103-parv/parkinsons-detection &nbsp;|&nbsp; joints are a canonical-skeleton "
           "approximation; validated on one public dataset; see the model card and clinical evaluation "
           "in the repo.", SMALL))

doc.build(E)
print(f"wrote {OUT}")
