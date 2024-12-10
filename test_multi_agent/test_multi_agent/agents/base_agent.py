# Create Agents

#base_agent.py

from abc import ABC, abstractmethod
from typing import Any, Dict
import ollama

class BaseAgent(ABC):
    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model_name = model_name
        
    async def get_completion(self, prompt: str) -> str:
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content']
        except Exception as e:
            raise Exception(f"Error getting completion: {str(e)}")

class MainAgent(BaseAgent):
    @abstractmethod
    async def process(self, input_data: Any) -> Dict[str, Any]:
        pass

class ValidatorAgent(BaseAgent):
    @abstractmethod
    async def validate(self, input_data: Any, output_data: Any) -> Dict[str, bool]:
        pass 