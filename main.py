from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pdf2image import convert_from_bytes
from openai import OpenAI
import io
import base64
import os
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# ایجاد کلاینت OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/analyze-resume-visual")
async def analyze_resume_visual(file: UploadFile = File(...), job_description: str = ""):
    try:
        pdf_bytes = await file.read()

        # تبدیل به تصویر - همه صفحات (یا فقط چند صفحه اول)
        images = convert_from_bytes(pdf_bytes, dpi=200)
        if not images:
            return JSONResponse(content={"error": "PDF conversion failed"}, status_code=400)

        # فقط ۳ صفحه اول رو بررسی کن (اگه کمتر بود، همونا)
        max_pages = min(3, len(images))
        image_messages = []

        for i in range(max_pages):
            buffered = io.BytesIO()
            images[i].save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
            image_messages.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_base64}"}
            })

        # اضافه کردن پرامپت جدید
        prompt = f"""
        You are a resume expert. Please analyze the following resume in detail. I want you to provide a comprehensive review, Please respond in the following numbered format exactly:

        1. **Match Percentage:** Please provide a percentage from 0 to 100 for how well this resume aligns with the job description I will provide. The analysis should be logical, not based on string matching, but on the relevance of skills, experience, and overall qualifications. Provide an explanation of your reasoning for this percentage.

        2. **Alternative Job Matches:** Based on the skills, experience, and qualifications presented in this resume, suggest other job roles or industries that this resume could be a good fit for. 

        3. **Resume Section Review:** Look through each section of the resume and analyze if there are any changes or improvements needed. For example, check if any sections lack clarity, if there's redundant information, or if there are any critical areas missing. Provide recommendations for improvement.

        4. **Writing Errors or Repetitions:** Scan for any spelling mistakes, grammatical errors, or repeated phrases. If there are any errors or unusual repetition, highlight them and suggest improvements.

        I will also provide the **job description** that this resume is being compared against.

        === Job Description ===
        {job_description}

        === Resume ===
        {resume_text[:6000]}  # Limiting to the first 6000 characters
        """

        # ارسال به GPT-4o با همه تصاویر با متد جدید
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": prompt}],
            max_tokens=800
        )

        feedback = response.choices[0].message.content
        return {"success": True, "feedback": feedback}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
