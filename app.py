import os
import time
import json
import asyncio
import random
from datetime import datetime
import gradio as gr

# Optional: use nested asyncio in some environments
try:
    import nest_asyncio
    nest_asyncio.apply()
except Exception:
    pass

# Try to load Gemini API key from environment (set in Spaces secrets)
USE_GEMINI = bool(os.environ.get("GOOGLE_API_KEY"))

# Try import gemini SDK only if available/needed
if USE_GEMINI:
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        LLM_AVAILABLE = True
    except Exception as e:
        print("Warning: google-generativeai failed to import/configure:", e)
        LLM_AVAILABLE = False
else:
    LLM_AVAILABLE = False

# Observability stores
LOGS = []
TRACES = []
METRICS = {"llm_calls": [], "latency": []}
MEMORY = []

def now_ts():
    return datetime.utcnow().isoformat()

def log(msg, agent=None, level="INFO"):
    entry = {"ts": now_ts(), "agent": agent, "level": level, "msg": msg}
    LOGS.append(entry)
    print(f"[{entry['ts']}] [{level}] [{agent}] {msg}")

def trace(step, data=None):
    entry = {"ts": now_ts(), "step": step, "data": data}
    TRACES.append(entry)
    print(f"[TRACE] {step} - {data}")

def metric(name, value):
    if name not in METRICS:
        METRICS[name] = []
    METRICS[name].append({"ts": now_ts(), "value": value})

# Simple Gemini helper
def gemini_call(prompt: str):
    start = time.perf_counter()
    metric("llm_calls", 1)
    if not LLM_AVAILABLE:
        # deterministic simulated responses (vary by prompt content)
        out = "(SIMULATED) " + prompt[:120]
    else:
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(prompt)
            out = getattr(resp, "text", str(resp))
        except Exception as e:
            log(f"Gemini error: {e}", agent="LLM", level="ERROR")
            out = f"(ERROR) {e}"
    elapsed = time.perf_counter() - start
    metric("latency", elapsed)
    return out

# Tools (stubs)
class CalendarTool:
    async def check_availability(self, prefs):
        await asyncio.sleep(0.05)
        if prefs:
            available = [p for p in prefs if random.random()>0.2]
            return available or ["2025-11-20T09:00"]
        return ["2025-11-20T09:00","2025-11-20T11:00"]

class NotesTool:
    async def summarize(self, text):
        await asyncio.sleep(0.03)
        return {"summary": text.split(".")[0]}

class SearchTool:
    async def query(self,q):
        await asyncio.sleep(0.02)
        return [{"title": f"Result for {q}", "link":"https://example.com"}]

# Memory bank simple
def save_memory(user, data):
    MEMORY.append({"user": user, "data": data, "ts": now_ts()})

def get_memory(user):
    return [m for m in MEMORY if m["user"]==user]

# Agents (async but we'll run via asyncio.run)
class TaskCreatorAgent:
    def __init__(self): pass
    async def run(self, user, text):
        log("TaskCreator running", agent="TaskCreator")
        title = gemini_call(f"Create a short task title for: {text}")
        task = {"title": title.strip()[:120], "description": text, "user": user, "preferred_times": []}
        save_memory(user, {"action":"create","task":task})
        trace("task.created", task)
        return task

class SummarizerAgent:
    def __init__(self, notes_tool): self.notes = notes_tool
    async def run(self, user, text):
        log("Summarizer running", agent="Summarizer")
        if LLM_AVAILABLE:
            summary = gemini_call(f"Summarize concisely: {text}")
            res = {"summary": summary}
        else:
            res = await self.notes.summarize(text)
        save_memory(user, {"action":"summarize","summary":res})
        trace("summary.generated", res)
        return res

class PrioritizerAgent:
    def __init__(self): pass
    async def run(self, task):
        log("Prioritizer running", agent="Prioritizer")
        if LLM_AVAILABLE:
            resp = gemini_call(f"Rate priority 0-100 for: {task['description']}. Reply only with the number.")
            try:
                score = float(''.join([c for c in resp if (c.isdigit() or c=='.')]))
            except:
                score = 50.0
        else:
            desc = task.get("description","").lower()
            score = 0
            if "urgent" in desc: score += 70
            if "today" in desc or "tomorrow" in desc: score += 30
        task["priority_score"] = score
        task["priority"] = "high" if score>70 else ("medium" if score>30 else "low")
        save_memory(task["user"], {"action":"prioritize","priority":task["priority"], "score": score})
        trace("task.prioritized", {"score":score})
        return task

class SchedulerAgent:
    def __init__(self, calendar): self.calendar = calendar
    async def run(self, task):
        log("Scheduler running", agent="Scheduler")
        slots = await self.calendar.check_availability(task.get("preferred_times",[]))
        if task["priority"]=="high":
            chosen = sorted(slots)[0]
        elif task["priority"]=="medium":
            chosen = slots[len(slots)//2]
        else:
            chosen = slots[-1]
        task["booking"] = {"slot": chosen, "status":"booked"}
        save_memory(task["user"], {"action":"schedule","slot":chosen})
        trace("task.scheduled", {"slot":chosen})
        return task

class ReminderAgent:
    async def run(self, task):
        log("Reminder scheduling", agent="Reminder")
        save_memory(task["user"], {"action":"reminder_scheduled","slot": task.get("booking",{}).get("slot")})
        trace("reminder.scheduled", {"slot": task.get("booking",{}).get("slot")})
        return task

# Orchestrator
async def run_pipeline_async(user, text, preferred_times=None):
    trace("pipeline.start", {"user":user})
    cal = CalendarTool()
    notes = NotesTool()
    creator = TaskCreatorAgent()
    summarizer = SummarizerAgent(notes)
    prioritizer = PrioritizerAgent()
    scheduler = SchedulerAgent(cal)
    reminder = ReminderAgent()

    task = await creator.run(user, text)
    if preferred_times:
        task["preferred_times"] = preferred_times
    _ = await summarizer.run(user, text)
    task = await prioritizer.run(task)
    task = await scheduler.run(task)
    task = await reminder.run(task)
    trace("pipeline.end", {"user":user, "title": task.get("title")})
    return task

def run_pipeline(user, text, preferred_times=None):
    # Synchronous wrapper for Gradio
    return asyncio.get_event_loop().run_until_complete(run_pipeline_async(user, text, preferred_times))

# Gradio UI
with gr.Blocks(title="Task Simplifier - Multi-Agent Demo") as demo:
    gr.Markdown("# Task Simplifier — Multi-Agent Demo\nThis Gradio app runs the multi-agent pipeline server-side and returns structured JSON output.\n\n**Note:** Set your `GOOGLE_API_KEY` in Space secrets to enable Gemini.")

    with gr.Row():
        inp = gr.Textbox(label="Task input", value="Pay electricity bill by Friday, urgent", lines=3)
        prefs = gr.Textbox(label="Preferred times (comma-separated ISO)", value="", lines=1)
    run_btn = gr.Button("Run Pipeline")
    output = gr.JSON(label="Output (task object)")
    logs_box = gr.Textbox(label="Recent Logs (last 20)", interactive=False)
    traces_box = gr.Textbox(label="Recent Traces (last 20)", interactive=False)
    metrics_box = gr.JSON(label="Metrics")

    def on_run(task_text, pref_text):
        prefs_list = [p.strip() for p in pref_text.split(",") if p.strip()]
        try:
            res = run_pipeline("hf_user", task_text, preferred_times=prefs_list or None)
        except Exception as e:
            log(f"Pipeline exception: {e}", agent="Orchestrator", level="ERROR")
            return {"error": str(e)}, "\n".join([str(l) for l in LOGS[-20:]]), "\n".join([str(t) for t in TRACES[-20:]]), METRICS
        out_json = res
        recent_logs = "\n".join([f"[{l['ts']}] {l['agent']}: {l['msg']}" for l in LOGS[-20:]])
        recent_traces = "\n".join([f"[{t['ts']}] {t['step']}: {t.get('data')}" for t in TRACES[-20:]])
        return out_json, recent_logs, recent_traces, METRICS

    run_btn.click(on_run, inputs=[inp, prefs], outputs=[output, logs_box, traces_box, metrics_box])

    gr.Markdown("----\n**Observability**: Logs, traces, metrics are kept in-memory for demo purposes. Use `export` in a real server for persistence.")

demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
