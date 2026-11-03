import io
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
)

from app.utils import format_class_name, format_confidence


SEVERITY_COLORS = {
    "none": colors.HexColor("#2e7d32"),
    "low": colors.HexColor("#9e9d24"),
    "moderate": colors.HexColor("#ef6c00"),
    "high": colors.HexColor("#d84315"),
    "severe": colors.HexColor("#b71c1c"),
    "unknown": colors.HexColor("#616161"),
}


def build_report(uploaded_file, result):
    """
    Build a PDF report for a single CropShield prediction.

    Args:
        uploaded_file: the Streamlit UploadedFile (already analyzed).
        result: the result dict returned by CropShieldInference.predict().

    Returns:
        BytesIO buffer containing the PDF, ready for st.download_button.
    """

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CSTitle",
        parent=styles["Title"],
        textColor=colors.HexColor("#1b3a1f"),
        fontSize=22,
    )
    heading_style = ParagraphStyle(
        "CSHeading",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#2e7d32"),
        spaceBefore=14,
        spaceAfter=6,
    )
    body_style = styles["Normal"]
    caption_style = ParagraphStyle(
        "CSCaption",
        parent=styles["Normal"],
        textColor=colors.HexColor("#666666"),
        fontSize=9,
    )

    story = []

    # ---- Header ----
    story.append(Paragraph("CropShield Diagnosis Report", title_style))
    story.append(
        Paragraph(
            datetime.now().strftime("Generated on %B %d, %Y at %I:%M %p"),
            caption_style,
        )
    )
    story.append(Spacer(1, 16))

    # ---- Image ----
    uploaded_file.seek(0)
    try:
        img = Image(uploaded_file, width=3.2 * inch, height=3.2 * inch)
        img.hAlign = "CENTER"
        story.append(img)
        story.append(Spacer(1, 16))
    except Exception:
        pass

    status = result.get("status")

    if status == "not_leaf":
        story.append(Paragraph("Result: No Leaf Detected", heading_style))
        story.append(
            Paragraph(
                "The uploaded image did not appear to contain a plant leaf.",
                body_style,
            )
        )
        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                f"Leaf probability: {format_confidence(result['leaf_probability'])}",
                body_style,
            )
        )
        doc.build(story)
        buffer.seek(0)
        return buffer

    # ---- Diagnosis summary table ----
    story.append(Paragraph("Diagnosis Summary", heading_style))

    predicted_class = format_class_name(result.get("predicted_class", "Unknown"))
    confidence = format_confidence(result.get("confidence", 0))
    leaf_probability = format_confidence(result.get("leaf_probability", 0))

    summary_data = [
        ["Predicted Class", predicted_class],
        ["Confidence", confidence],
        ["Leaf Probability", leaf_probability],
        ["Status", "Low Confidence" if status == "low_confidence" else "Confirmed"],
    ]

    summary_table = Table(summary_data, colWidths=[2 * inch, 3.5 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#e8f5e9")),
                ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#1b3a1f")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(summary_table)
    story.append(Spacer(1, 10))

    # ---- Top 3 predictions ----
    top_3 = result.get("top_3")
    if top_3:
        story.append(Paragraph("Top 3 Predictions", heading_style))
        top3_data = [["Class", "Confidence"]]
        for p in top_3:
            top3_data.append(
                [format_class_name(p["class_name"]), format_confidence(p["confidence"])]
            )
        top3_table = Table(top3_data, colWidths=[4 * inch, 1.5 * inch])
        top3_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e7d32")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dddddd")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ]
            )
        )
        story.append(top3_table)
        story.append(Spacer(1, 10))

    # ---- Treatment info ----
    treatment_info = result.get("treatment_info")
    if treatment_info:
        story.append(Paragraph("Treatment &amp; Prevention", heading_style))

        severity = treatment_info.get("severity", "unknown")
        sev_color = SEVERITY_COLORS.get(severity, SEVERITY_COLORS["unknown"])
        sev_style = ParagraphStyle(
            "CSSeverity",
            parent=body_style,
            textColor=sev_color,
            fontName="Helvetica-Bold",
        )
        story.append(Paragraph(f"Severity: {severity.title()}", sev_style))
        story.append(Spacer(1, 6))

        story.append(Paragraph("<b>Recommended treatment:</b>", body_style))
        for tip in treatment_info.get("treatment", []):
            story.append(Paragraph(f"&bull; {tip}", body_style))

        prevention = treatment_info.get("prevention", [])
        if prevention:
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Prevention tips:</b>", body_style))
            for tip in prevention:
                story.append(Paragraph(f"&bull; {tip}", body_style))

    story.append(Spacer(1, 20))
    story.append(
        Paragraph(
            "This report was generated automatically by CropShield and is "
            "intended as a decision-support aid, not a substitute for "
            "professional agronomic advice.",
            caption_style,
        )
    )

    doc.build(story)
    buffer.seek(0)
    return buffer