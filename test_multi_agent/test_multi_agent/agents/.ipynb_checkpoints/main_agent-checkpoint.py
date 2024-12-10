# main_agent.py

from typing import Any, Dict
from .base_agent import MainAgent

class SummarizeAgent(MainAgent):
    async def process(self, input_data: str) -> Dict[str, Any]:
        prompt = f"Summarize the following medical text:\n\n{input_data}"
        summary = await self.get_completion(prompt)
        return {"summary": summary}

class WriteArticleAgent(MainAgent):
    async def process(self, input_data: Dict[str, str]) -> Dict[str, Any]:
        prompt = f"""Write a research article with the following:
        Topic: {input_data['topic']}
        Key points: {input_data['key_points']}"""
        article = await self.get_completion(prompt)
        return {"article": article}

class SanitizeDataAgent(MainAgent):
    async def process(self, input_data: str) -> Dict[str, Any]:
        prompt = """Mask all Protected Health Information (PHI) in the following text. 
        Replace with appropriate masks:
        - Patient names with [PATIENT_NAME]
        - Doctor/Provider names with [PROVIDER_NAME]
        - Dates with [DATE]
        - Locations/Addresses with [LOCATION]
        - Phone numbers with [PHONE]
        - Email addresses with [EMAIL]
        - Medical record numbers with [MRN]
        - Social Security numbers with [SSN]
        - Device identifiers with [DEVICE_ID]
        - Any other identifying numbers with [ID]
        - Physical health conditions with [HEALTH_CONDITION]
        - Medications with [MEDICATION]
        - Lab results with [LAB_RESULT]
        - Vital signs with [VITAL_SIGN]
        - Procedures with [PROCEDURE]

        Text to mask:\n\n""" + input_data
        sanitized_data = await self.get_completion(prompt)
        return {"sanitized_data": sanitized_data} 