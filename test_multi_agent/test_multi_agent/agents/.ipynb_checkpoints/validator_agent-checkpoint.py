# validator_agent.py


from typing import Any, Dict
from .base_agent import ValidatorAgent

class SummarizeValidatorAgent(ValidatorAgent):
    async def validate(self, input_data: str, output_data: Dict[str, Any]) -> Dict[str, bool]:
        prompt = f"""Evaluate if this summary accurately represents the original text:
        Original: {input_data}
        Summary: {output_data['summary']}
        
        Provide:
        1. A score out of 5 (where 5 is perfect)
        2. 'valid' or 'invalid'
        3. Brief explanation
        
        Format: Score: X/5\nStatus: valid/invalid\nExplanation: ..."""
        
        result = await self.get_completion(prompt)
        is_valid = "valid" in result.lower()
        return {"is_valid": is_valid, "feedback": result}

class RefinerAgent(ValidatorAgent):
    async def validate(self, input_data: Dict[str, str], output_data: Dict[str, Any]) -> Dict[str, bool]:
        prompt = f"""Review this research article for quality and accuracy:
        Article: {output_data['article']}
        
        Provide:
        1. A score out of 5 (where 5 is perfect)
        2. 'valid' or 'invalid'
        3. Brief explanation
        
        Format: Score: X/5\nStatus: valid/invalid\nExplanation: ..."""
        
        result = await self.get_completion(prompt)
        is_valid = "valid" in result.lower()
        return {"is_valid": is_valid, "feedback": result}

class SanitizeValidatorAgent(ValidatorAgent):
    async def validate(self, input_data: str, output_data: Dict[str, Any]) -> Dict[str, bool]:
        prompt = f"""Verify if all Protected Health Information (PHI) has been properly masked in this text:
        Masked text: {output_data['sanitized_data']}
        
        Check for any unmasked:
        - Patient names
        - Doctor/Provider names
        - Dates
        - Locations/Addresses
        - Phone numbers
        - Email addresses
        - Medical record numbers
        - Social Security numbers
        - Device identifiers
        - Other identifying numbers
        - Physical health conditions
        - Medications
        - Lab results
        - Vital signs
        - Procedures
        
        Provide:
        1. A score out of 5 (where 5 means all PHI properly masked)
        2. 'valid' or 'invalid'
        3. List any found unmasked PHI
        
        Format: Score: X/5\nStatus: valid/invalid\nFindings: ..."""
        
        result = await self.get_completion(prompt)
        is_valid = "valid" in result.lower()
        return {"is_valid": is_valid, "feedback": result} 
