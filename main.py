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
                {"role": "system", "content": "You are a professional resume design expert. Please analyze the following resume images."},
                {"role": "user", "content": [
                    {"type": "text", "text": "You are a professional resume design expert. Please analyze the following resume images in detail. Comment on the following aspects:"},
                    {"type": "text", "text": "1. Match Percentage: Please provide a percentage from 0 to 100 for how well this resume aligns with the job description."},
                    {"type": "text", "text": "2. Alternative Job Matches: Suggest other job roles or industries that this resume could be a good fit for."},
                    {"type": "text", "text": "3. Resume Section Review: Analyze the sections of the resume and suggest any improvements or missing sections."},
                    {"type": "text", "text": "4. Writing Errors or Repetitions: Check for any spelling mistakes, grammatical errors, or repetitive phrases."},
                    *image_messages
                ]}
            ],
            max_tokens=800
        )

        feedback = response.choices[0].message.content
        return {"success": True, "feedback": feedback}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
