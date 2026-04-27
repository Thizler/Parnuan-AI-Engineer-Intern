import os
import json
import re 
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from dotenv import load_dotenv

# โหลด Environment Variables
base_path = Path(__file__).resolve().parent.parent
env_path = base_path / ".env"
load_dotenv(dotenv_path=env_path)

class Transaction(BaseModel):
    """Schema สำหรับ 1 รายการธุรกรรม"""
    amount: float = Field(..., description="The monetary value, numeric only")
    detail: str = Field(..., description="What the money was spent on")

class TransactionResponse(BaseModel):
    """Schema สำหรับการตอบกลับที่อาจมีหลายรายการ"""
    transactions: List[Transaction] = Field(default_factory=list)

class NERSystem:
    def __init__(self, model_name: str = "anthropic/claude-haiku-4.5"):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("❌ ไม่พบ OPENROUTER_API_KEY ในไฟล์ .env")
            
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name

    def get_system_prompt(self):
        """กำหนดกฎเกณฑ์ให้ LLM"""
        return """You are a Thai Transaction NER assistant. 
        Extract transactions from text. Return ONLY a JSON object with the key 'transactions'.
        Rules:
        1. amount: numeric only.
        2. detail: description of item.
        3. If no transaction, return {"transactions": []}.
        4. Do not include any explanations, only the JSON."""

    def parse_with_regex(self, text: str):
        """[Bonus] Cost Optimization ด้วย Regex"""
        pattern = r"^([\u0E00-\u0E7Fa-zA-Z\s]+?)\s+(\d+(?:\.\d+)?)\s*(?:บาท|฿)?$"
        match = re.match(pattern, text.strip())
        if match:
            detail = match.group(1).strip()
            amount = float(match.group(2))
            return TransactionResponse(transactions=[Transaction(amount=amount, detail=detail)])
        return None

    def parse(self, text: str) -> TransactionResponse:
        """แปลงข้อความดิบ (แก้ไขเพื่อรองรับ Claude 4.5)"""
        regex_result = self.parse_with_regex(text)
        if regex_result:
            return regex_result

        try:
            # ถอด response_format ออกเพื่อให้ Claude 4.5 ทำงานได้เสถียรขึ้น
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.get_system_prompt()},
                    {"role": "user", "content": text}
                ]
            )
            
            raw_content = response.choices[0].message.content
            
            # ระบบแกะ JSON: ค้นหาข้อความระหว่าง { และ } เพื่อจัดการ Markdown
            json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
            clean_json = json_match.group(0) if json_match else raw_content
            
            data = json.loads(clean_json)
            return TransactionResponse(**data)
            
        except Exception as e:
            print(f"\n❌ Error with {self.model_name}: {e}")
            return TransactionResponse(transactions=[])