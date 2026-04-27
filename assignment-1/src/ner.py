import os
import json
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# บังคับโหลดไฟล์ .env จากโฟลเดอร์หลักของโปรเจค
base_path = Path(__file__).resolve().parent.parent
env_path = base_path / ".env"
load_dotenv(dotenv_path=env_path)

class Transaction(BaseModel):
    """Schema สำหรับ 1 รายการธุรกรรมตามที่โจทย์กำหนด"""
    amount: float = Field(..., description="The monetary value, numeric only")
    detail: str = Field(..., description="What the money was spent on")

class TransactionResponse(BaseModel):
    """Schema สำหรับการตอบกลับของระบบที่อาจมีได้หลายรายการ"""
    transactions: List[Transaction] = Field(default_factory=list)

class NERSystem:
    def __init__(self, model_name: str = "google/gemini-2.5-flash"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("❌ ไม่พบ OPENROUTER_API_KEY ในไฟล์ .env")
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name

    def get_system_prompt(self):
        """กำหนดบทบาทและกฎเกณฑ์ให้ LLM เพื่อความแม่นยำสูงสุด"""
        return """You are a Thai Transaction NER assistant. 
        Extract transactions from free-form Thai or mixed Thai/English text. 
        Rules:
        1. amount: numeric only (no currency symbols like บาท, ฿).
        2. detail: exact merchant, item, or service description.
        3. If multiple items are in one message, extract all separately.
        4. If the message is a greeting or contains NO transaction, return an empty list.
        5. NEVER hallucinate or invent data not present in the text.
        6. Always return a JSON object with the key 'transactions'."""

    def parse(self, text: str) -> TransactionResponse:
        """แปลงข้อความดิบให้เป็นโครงสร้างข้อมูล JSON"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": text}
                ],
                response_format={ "type": "json_object" }
            )
            
            raw_content = response.choices[0].message.content
            data = json.loads(raw_content)
            
            # ตรวจสอบความถูกต้องของ Schema ด้วย Pydantic
            return TransactionResponse(**data)
            
        except Exception as e:
            # กลไก Graceful Degradation: หากพัง ให้คืนค่าว่างเสมอเพื่อไม่ให้ระบบล่ม
            print(f"\n❌ Error parsing text '{text[:20]}...': {e}")
            return TransactionResponse(transactions=[])