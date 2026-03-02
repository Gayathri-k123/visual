from fpdf import FPDF

def generate_pdf_bytes(username, email, date_str, time_str, score, report_id):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Student Engagement Report", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Student Name: {username}", ln=True)
    pdf.cell(200, 10, txt=f"Email: {email}", ln=True)
    pdf.cell(200, 10, txt=f"Date: {date_str}", ln=True)
    pdf.cell(200, 10, txt=f"Time: {time_str}", ln=True)
    pdf.ln(10)
    
    # --- IF CHEATED (Score is 0) ---
    if score == 0:
        pdf.set_text_color(255, 0, 0) # Red color
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(200, 10, txt="RESULT: DISQUALIFIED", ln=True)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Reason: Tab Switching / Window Violation Detected.", ln=True)
        pdf.ln(5)
        pdf.set_text_color(0, 0, 0) # Back to Black
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=f"Final Score: {score}%", ln=True)
    
    # --- NORMAL SESSION ---
    else:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=f"Engagement Score: {score}%", ln=True)
        pdf.set_font("Arial", 'I', 10)
        if score > 75:
            pdf.cell(200, 10, txt="Status: Excellent Focus", ln=True)
        elif score > 50:
            pdf.cell(200, 10, txt="Status: Moderate Attention", ln=True)
        else:
            pdf.cell(200, 10, txt="Status: Needs Improvement", ln=True)

    pdf.ln(20)
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt="* This report was generated automatically by the AI Proctoring System.", ln=True)

    # --- THIS FIXES THE CRASH ---
    out = pdf.output(dest='S')
    if isinstance(out, str):
        return out.encode('latin-1')
    return bytes(out)