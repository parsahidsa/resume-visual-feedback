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
async def analyze_resume_visual(file: UploadFile = File(...)):
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

        # ارسال به GPT-4o با همه تصاویر با متد جدید
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a professional resume expert. Please analyze the following resume in detail, providing a comprehensive review."},
                {"role": "user", "content": f"Please respond in the following numbered format exactly:\n"
                                           f"1. Match Percentage: Provide a match percentage from 0 to 100 for how well this resume aligns with the job description.\n"
                                           f"2. Alternative Job Matches: Suggest other roles or industries this resume could fit.\n"
                                           f"3. Design & Layout: Is the resume visually appealing? Comment on alignment, font, spacing, and clarity.\n"
                                           f"4. Grammar & Phrasing: Highlight any errors, awkward phrases, or redundancies.\n"
                                           f"5. Suggestions for Improvement: Provide overall recommendations for improving the resume in terms of content and design.\n"
                                           f"6. Profile Photo: If a photo is included, is it professionally presented?\n"
                                           f"Provide detailed insights, including any issues with the visual design, such as excessive color use, poor contrast, or poorly formatted images."},
                {"role": "user", "content": f"Job Description: {job_description}"},
                *image_messages
            ],
            max_tokens=8000
        )

        feedback = response.choices[0].message.content
        return {"success": True, "feedback": feedback}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
