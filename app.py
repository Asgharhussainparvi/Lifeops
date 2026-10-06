import os
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)


class DecisionAnalysis(BaseModel):
    goal: str
    objective: str
    constraints: list[str]
    assumptions: list[str]
    recommendation: str
    reasoning: str
    risks: list[str]
    next_steps: list[str]


goal = input("What decision do you want help with? ")

prompt = f"""
You are a decision analysis assistant.

Analyze the user's goal carefully.

User goal:
{goal}

Be practical and concise.
Do not invent facts.
If important information is missing, clearly state the assumption.
"""


response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt,
    config={
        "response_mime_type": "application/json",
        "response_schema": DecisionAnalysis,
    },
)

print("\n===== LIFEOPS DECISION =====\n")
print(response.text)
