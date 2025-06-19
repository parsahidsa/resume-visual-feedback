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
                {"role": "system", "content": "You are a resume design advisor. Give visual feedback on the uploaded resume images."},
                {"role": "user", "content": [
                    {"type": "text", "text": "Please analyze this resume visually. Comment on profile photo, layout, readability, and design."},
                    *image_messages
                ]}
            ],
            max_tokens=800
        )

        feedback = response.choices[0].message.content
        return {"success": True, "feedback": feedback}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
