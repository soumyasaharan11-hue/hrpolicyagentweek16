import os
import re
import time
from dotenv import load_dotenv

load_dotenv(override=True)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

HR_POLICIES = {
    "annual_leave": "Employees receive 24 days of annual leave per calendar year. Leave should normally be requested at least 5 working days in advance. Up to 5 unused annual leave days may be carried forward into the next year.",
    "sick_leave": "Employees should notify their manager as soon as reasonably possible when they are unable to work because of illness. A medical certificate may be required for absences longer than 3 consecutive working days.",
    "work_from_home": "Employees may work remotely for up to 2 days per week, subject to team requirements and manager approval.",
    "parental_leave": "Eligible employees may request parental leave in accordance with company policy and applicable local employment law. Employees should contact HR for current eligibility and documentation requirements.",
    "maternity_leave": "Eligible employees may take maternity leave according to the current company policy and applicable local employment law.",
    "paternity_leave": "Eligible employees may request paternity leave under the current company policy and applicable local employment law.",
    "bereavement_leave": "Employees may request bereavement leave following the death of an immediate family member. The exact entitlement depends on current company policy and applicable law.",
    "working_hours": "Standard working hours are 9:00 AM to 6:00 PM, Monday to Friday, with a one-hour lunch break.",
    "overtime": "Overtime should be agreed with the employee's manager before it is worked. Compensation or time off in lieu follows applicable company policy and local employment requirements.",
    "performance_review": "Performance reviews are conducted twice a year.",
    "probation": "New employees normally complete a 6-month probation period. The employment contract determines specific individual terms.",
    "notice_period": "The standard notice period is 30 days unless the employee's contract specifies a different period or applicable law requires otherwise.",
    "code_of_conduct": "Employees are expected to behave professionally, treat colleagues with respect, and comply with anti-harassment, anti-discrimination, confidentiality, and workplace conduct rules.",
    "expenses": "Reasonable business expenses should be submitted through the approved expense process with supporting receipts. Approval is required before reimbursement is processed.",
}

@tool
def search_hr_policy(query: str) -> str:
    """Search the HR policy knowledge base."""
    query_lower = query.lower()
    matches = []

    for key, content in HR_POLICIES.items():
        haystack = f"{key} {content}".lower()
        keywords = [w for w in re.findall(r"\w+", query_lower) if len(w) > 2]
        score = sum(1 for word in keywords if word in haystack)
        if score:
            matches.append((score, key, content))

    if not matches:
        return (
            "No matching HR policy was found. Do not invent an answer. "
            "Tell the employee to contact HR."
        )

    matches.sort(reverse=True)
    return "\n\n".join(
        f"{key}: {content}" for _, key, content in matches[:3]
    )

@tool
def list_hr_policy_categories() -> str:
    """List available HR policy categories."""
    return "\n".join(f"- {key}" for key in HR_POLICIES)

SYSTEM_PROMPT = """
You are an HR Policy Assistant.
Use the HR policy tools to answer employee questions.
Never invent a company policy.
If no matching policy is available, tell the employee to contact HR.
Do not expose confidential employee information.
Do not make employee-specific legal or eligibility determinations.
Be concise, professional, and helpful.
"""

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)

memory = MemorySaver()

agent = create_react_agent(
    model=llm,
    tools=[search_hr_policy, list_hr_policy_categories],
    checkpointer=memory,
    prompt=SYSTEM_PROMPT,
)

app = FastAPI(
    title="HR Policy AI Agent",
    description="LangGraph-powered HR policy chatbot",
    version="1.0.0",
)

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique conversation session ID")
    message: str = Field(..., description="Employee's HR policy question")

class ChatResponse(BaseModel):
    session_id: str
    response: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    model: str
    tools_available: list

SERVER_START_TIME = time.time()

@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(time.time() - SERVER_START_TIME, 2),
        model="gpt-4o-mini",
        tools_available=[
            "search_hr_policy",
            "list_hr_policy_categories",
        ],
    )

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    start_time = time.time()

    try:
        config = {
            "configurable": {
                "thread_id": request.session_id
            }
        }

        result = agent.invoke(
            {"messages": [HumanMessage(content=request.message)]},
            config=config,
        )

        response_text = result["messages"][-1].content
        latency_ms = round((time.time() - start_time) * 1000, 2)

        return ChatResponse(
            session_id=request.session_id,
            response=response_text,
            latency_ms=latency_ms,
        )

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="HR Policy Agent is temporarily unavailable. Please try again."
        )
