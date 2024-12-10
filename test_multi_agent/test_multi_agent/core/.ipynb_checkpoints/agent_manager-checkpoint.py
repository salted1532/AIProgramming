# Core Functions

# agent_manager.py

from typing import Dict, Any
from agents.main_agent import SummarizeAgent, WriteArticleAgent, SanitizeDataAgent
from agents.validator_agent import SummarizeValidatorAgent, RefinerAgent, SanitizeValidatorAgent
from core.logger import Logger

class AgentManager:
    def __init__(self):
        self.logger = Logger()
        
        # Initialize main agents
        self.summarize_agent = SummarizeAgent()
        self.write_article_agent = WriteArticleAgent()
        self.sanitize_agent = SanitizeDataAgent()
        
        # Initialize validator agents
        self.summarize_validator = SummarizeValidatorAgent()
        self.refiner_agent = RefinerAgent()
        self.sanitize_validator = SanitizeValidatorAgent()

    async def process_task(self, task_type: str, input_data: Any) -> Dict[str, Any]:
        try:
            self.logger.log_input(task_type, input_data)
            
            if task_type == "summarize":
                result = await self.summarize_agent.process(input_data)
                validation = await self.summarize_validator.validate(input_data, result)
            
            elif task_type == "write_article":
                result = await self.write_article_agent.process(input_data)
                validation = await self.refiner_agent.validate(input_data, result)
            
            elif task_type == "sanitize":
                result = await self.sanitize_agent.process(input_data)
                validation = await self.sanitize_validator.validate(input_data, result)
            else:
                raise ValueError(f"Unknown task type: {task_type}")

            self.logger.log_output(task_type, result, validation)
            return {"result": result, "validation": validation}

        except Exception as e:
            self.logger.log_error(task_type, str(e))
            raise 