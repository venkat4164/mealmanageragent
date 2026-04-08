import os
import logging
import datetime
import google.cloud.logging
from google.cloud import firestore
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from mcp.server.fastmcp import FastMCP 

from google.adk import Agent
from google.adk.agents import SequentialAgent

# --- 1. Logging ---
try:
    cloud_logging_client = google.cloud.logging.Client()
    cloud_logging_client.setup_logging()
except Exception:
    logging.basicConfig(level=logging.INFO)

load_dotenv()
model_name = os.getenv("MODEL", "gemini-1.5-flash")

# --- 2. Firestore ---
db = firestore.Client(database="mealdb")

mcp = FastMCP("MealManagerTools")

# ================= TOOLS =================

@mcp.tool()
def add_task(title: str) -> str:
    try:
        db.collection("tasks").add({
            "title": title,
            "status": "pending",
            "created_at": datetime.datetime.utcnow()
        })
        return f"Task added: {title}"
    except Exception as e:
        return str(e)

@mcp.tool()
def list_tasks() -> str:
    try:
        tasks = db.collection("tasks").stream()
        res = ["📋 Tasks:"]
        for t in tasks:
            data = t.to_dict()
            status = "✅" if data.get("status") == "done" else "⏳"
            res.append(f"{status} {data.get('title')}")
        return "\n".join(res)
    except Exception as e:
        return str(e)

@mcp.tool()
def complete_task(task_title: str) -> str:
    try:
        tasks = db.collection("tasks").where("title", "==", task_title).stream()
        for t in tasks:
            db.collection("tasks").document(t.id).update({"status": "done"})
            return f"Task completed: {task_title}"
        return "Task not found"
    except Exception as e:
        return str(e)

@mcp.tool()
def save_meal_plan(plan: str) -> str:
    try:
        db.collection("meal_plans").add({
            "plan": plan,
            "created_at": datetime.datetime.utcnow()
        })
        return "Meal plan saved"
    except Exception as e:
        return str(e)

# ================= AGENTS =================

# 🍽️ Meal Agent
def meal_instruction(ctx):
    user_input = ctx.state.get("user_input", "")
    return f"""
You are a meal planning assistant.

User Input:
{user_input}

Tasks:
1. Identify ingredients
2. Generate:
   - Breakfast
   - Lunch
   - Dinner

Rules:
- Keep meals simple (prefer Indian food)
- Use available ingredients

Return STRICT JSON:
{{
 "breakfast": "",
 "lunch": "",
 "dinner": ""
}}
"""

meal_agent = Agent(
    name="meal_planner",
    model=model_name,
    instruction=meal_instruction,
    tools=[save_meal_plan],
    output_key="meal_data"
)

# 🛒 Grocery Agent
def grocery_instruction(ctx):
    meal_data = ctx.state.get("meal_data", "")
    return f"""
Generate a grocery list based on this meal plan.

MEAL PLAN:
{meal_data}

Rules:
- Only missing items
- Keep minimal
"""

grocery_agent = Agent(
    name="grocery_agent",
    model=model_name,
    instruction=grocery_instruction,
    output_key="grocery_data"
)

# ⏰ Scheduler Agent
def scheduler_instruction(ctx):
    grocery_data = ctx.state.get("grocery_data", "")
    return f"""
Create simple tasks based on grocery and meals.

GROCERY:
{grocery_data}

Examples:
- Buy groceries
- Cook meals

Use tool to save tasks.
"""

scheduler_agent = Agent(
    name="scheduler_agent",
    model=model_name,
    instruction=scheduler_instruction,
    tools=[add_task]
)

# 🔁 Workflow
meal_workflow = SequentialAgent(
    name="meal_workflow",
    sub_agents=[
        meal_agent,
        grocery_agent,
        scheduler_agent
    ]
)

# 🧠 Root Agent
def root_instruction(ctx):
    user_input = ctx.state.get("user_input", "")
    return f"""
You are MealManagerAgent.

User Input:
{user_input}

Execute full workflow:
1. Meal plan
2. Grocery list
3. Task creation

Return a friendly summary.
"""

root_agent = Agent(
    name="meal_manager_root",
    model=model_name,
    instruction=root_instruction,
    sub_agents=[meal_workflow]
)

# ================= API =================

app = FastAPI()

class UserRequest(BaseModel):
    prompt: str

@app.post("/api/v1/mealmanager/chat")
async def chat(request: UserRequest):
    try:
        final_reply = ""

        async for event in root_agent.run_async({
            "state": {
                "user_input": request.prompt
            }
        }):
            if hasattr(event, 'text') and event.text:
                final_reply = event.text

        return {
            "status": "success",
            "reply": final_reply or "Processed"
        }

    except Exception as e:
        logging.error(str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)