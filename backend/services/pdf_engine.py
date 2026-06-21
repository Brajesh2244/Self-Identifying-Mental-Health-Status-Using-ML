import os
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def generate_health_report(user_id: str, assessment_data: dict, action_plan: list) -> str:
    """
    Simulates generating a premium PDF Health Report and 7-Day Action Plan.
    Returns the file path to the generated PDF.
    """
    output_dir = "backend/demo_data/reports"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = f"{output_dir}/{user_id}_report.pdf"
    
    c = canvas.Canvas(file_path, pagesize=letter)
    
    # Premium Cover Page
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, 700, "Premium Health Intelligence Report")
    
    c.setFont("Helvetica", 14)
    c.drawString(100, 650, f"Risk Analysis: {assessment_data.get('risk_level', 'Moderate')}")
    
    # AI 7-Day Action Plan
    c.setFont("Helvetica-Bold", 18)
    c.drawString(100, 550, "Your 7-Day AI Action Plan:")
    
    c.setFont("Helvetica", 12)
    y_pos = 500
    for day, task in enumerate(action_plan, start=1):
        c.drawString(120, y_pos, f"Day {day}: {task}")
        y_pos -= 30
        
    c.save()
    
    return file_path
