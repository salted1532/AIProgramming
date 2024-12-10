# medical 분야의 챗봇

# https://medium.com/the-ai-forum/building-a-multi-agent-ai-system-from-scratch-for-medical-text-processing-dc6f10fc5f04

medical_ai_agents/
├── agents/
│ ├── __init__.py
│ ├── base_agent.py
│ ├── main_agents.py
│ └── validator_agents.py
├── core/
│ ├── __init__.py
│ ├── agent_manager.py
│ └── logger.py
├── utils/
│ ├── __init__.py
│ └── ollama_utils.py
├── app.py
└── requirements.txt

streamlit run app.py