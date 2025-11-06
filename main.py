"""
main.py - Multi-Agent Research Assistant API

This is the main entry point for a FastAPI web application that orchestrates
a multi-agent AI system to perform research tasks. The application:
1. Receives research queries from users
2. Creates a step-by-step research plan
3. Executes the plan using specialized AI agents (research, writer, editor)
4. Tracks progress in real-time
5. Returns comprehensive research reports

Architecture:
- FastAPI for the web server and REST API
- PostgreSQL for persistent storage of tasks and results
- Threading for non-blocking background execution
- In-memory dictionary for real-time progress tracking
"""

# ============================================================================
# IMPORTS - External Dependencies
# ============================================================================

# Standard library imports
import os           # For accessing environment variables and file system
import uuid         # For generating unique task identifiers
import json         # For serializing/deserializing JSON data
import threading    # For running agent workflows in background threads
from datetime import datetime  # For timestamping task creation and updates
from typing import Optional, Literal  # For type hints

# FastAPI framework imports
from fastapi import FastAPI, HTTPException, Request  # Core FastAPI components
from fastapi.responses import HTMLResponse, JSONResponse  # Response types
from fastapi.staticfiles import StaticFiles  # For serving CSS, JS, images
from fastapi.templating import Jinja2Templates  # For rendering HTML templates
from fastapi.middleware.cors import CORSMiddleware  # For cross-origin requests

# Data validation
from pydantic import BaseModel  # For request/response validation

# Database ORM imports
from sqlalchemy import create_engine, Column, Text, DateTime, String
from sqlalchemy.orm import sessionmaker, declarative_base

# Environment variable management
from dotenv import load_dotenv  # Loads API keys from .env file

# Our custom agent modules
from src.planning_agent import planner_agent, executor_agent_step

# HTML escaping utilities
import html, textwrap


# ============================================================================
# CONFIGURATION - Environment Setup
# ============================================================================

# Load environment variables from .env file (API keys, database URL, etc.)
load_dotenv()

# Get database connection string from environment
# Format: postgresql://username:password@host:port/database
DATABASE_URL = os.getenv("DATABASE_URL")

# Heroku uses "postgres://" but SQLAlchemy requires "postgresql://"
# This fixes compatibility issues when deploying to Heroku
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Validate that DATABASE_URL is set - the app cannot run without it
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set")


# ============================================================================
# DATABASE SETUP - SQLAlchemy Configuration
# ============================================================================

# Create the base class for our database models
Base = declarative_base()

# Create database engine
# - echo=False: Don't print SQL queries to console (set True for debugging)
# - future=True: Use SQLAlchemy 2.0 style API
engine = create_engine(DATABASE_URL, echo=False, future=True)

# Create session factory for database connections
# Each API call will create a new session to interact with the database
SessionLocal = sessionmaker(bind=engine)


# ============================================================================
# DATABASE MODELS - Task Table Definition
# ============================================================================

class Task(Base):
    """
    Task model represents a research task in the database.

    Each task goes through the following lifecycle:
    1. Created with status="running" when user submits a research query
    2. Background thread processes the task through multiple agent steps
    3. Final status is either "done" (with results) or "error"

    Attributes:
        id: Unique UUID identifier for tracking this task
        prompt: The user's original research query
        status: Current state - "running", "done", or "error"
        created_at: When the task was first submitted
        updated_at: When the task was last modified
        result: JSON string containing the final report and execution history
    """
    __tablename__ = "tasks"

    # Primary key - unique identifier for each task
    id = Column(String, primary_key=True, index=True)

    # User's research query (can be very long, hence Text type)
    prompt = Column(Text)

    # Current status of the task
    status = Column(String)

    # Timestamps for tracking task lifecycle
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

    # Final result stored as JSON string (contains report and step history)
    result = Column(Text)


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

# WARNING: This drops all tables on startup!
# In production, this should be controlled by an environment variable
# or removed entirely in favor of proper database migrations (e.g., Alembic)
try:
    Base.metadata.drop_all(bind=engine)
except Exception as e:
    print(f"❌ DB creation failed: {e}")

# Create all tables defined in our models
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"❌ DB creation failed: {e}")


# ============================================================================
# FASTAPI APPLICATION SETUP
# ============================================================================

# Initialize FastAPI application
app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing)
# WARNING: allow_origins=["*"] allows ANY website to call this API
# In production, restrict this to your specific domain(s)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Allow requests from any origin
    allow_methods=["*"],      # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"]       # Allow all headers
)

# Mount static files directory for serving CSS, JavaScript, images
# Files in /static will be accessible at /static/filename
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure Jinja2 template engine for rendering HTML pages
templates = Jinja2Templates(directory="templates")

# In-memory storage for real-time task progress
# Structure: {task_id: {"steps": [{"title": ..., "status": ..., "substeps": [...]}]}}
# WARNING: This data is lost on server restart. For production, store in database.
task_progress = {}


# ============================================================================
# REQUEST/RESPONSE MODELS - Data Validation
# ============================================================================

class PromptRequest(BaseModel):
    """
    Validates the incoming request body for /generate_report endpoint.

    Ensures the client sends a JSON object with a "prompt" field.
    Example: {"prompt": "Research AI applications in healthcare"}
    """
    prompt: str


# ============================================================================
# API ENDPOINTS - Route Handlers
# ============================================================================

@app.get("/", response_class=HTMLResponse)
def read_index(request: Request):
    """
    Serves the main web interface at the root URL.

    When a user visits http://localhost:8000/, this endpoint:
    1. Renders the templates/index.html file
    2. Returns it as an HTML page

    The index.html page provides a user-friendly interface for:
    - Entering research queries
    - Viewing step-by-step progress
    - Downloading final reports
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api", response_class=JSONResponse)
def health_check(request: Request):
    """
    Simple health check endpoint to verify the API is running.

    Returns: {"status": "ok"}

    Useful for:
    - Monitoring services to check if server is alive
    - Load balancers to verify the service is healthy
    - Quick manual tests during development
    """
    return {"status": "ok"}


@app.post("/generate_report")
def generate_report(req: PromptRequest):
    """
    Main endpoint for initiating a new research task.

    This is where the magic begins! When a user submits a research query,
    this endpoint:

    1. Creates a unique task ID (UUID)
    2. Saves the task to database with status="running"
    3. Calls planner_agent() to create a step-by-step research plan
    4. Initializes progress tracking with all planned steps
    5. Launches a background thread to execute the workflow
    6. Immediately returns the task_id to the user (non-blocking)

    The user can then poll /task_progress/{task_id} to watch progress
    and /task_status/{task_id} to get the final results.

    Args:
        req: PromptRequest object containing the research query

    Returns:
        {"task_id": "uuid-string"} - Unique identifier for tracking this task

    Example:
        POST /generate_report
        Body: {"prompt": "Large Language Models in drug discovery"}
        Response: {"task_id": "abc-123-def-456"}
    """
    # Generate a unique identifier for this task
    task_id = str(uuid.uuid4())

    # Save task to database with initial status
    db = SessionLocal()
    db.add(Task(id=task_id, prompt=req.prompt, status="running"))
    db.commit()
    db.close()

    # Initialize in-memory progress tracking
    task_progress[task_id] = {"steps": []}

    # Call planner_agent to create a step-by-step plan
    # Example output: [
    #   "Research agent: Search Tavily for relevant papers",
    #   "Research agent: Search arXiv for academic sources",
    #   "Writer agent: Draft comprehensive report",
    #   "Editor agent: Review and polish the report"
    # ]
    initial_plan_steps = planner_agent(req.prompt)

    # Populate progress tracker with all planned steps
    for step_title in initial_plan_steps:
        task_progress[task_id]["steps"].append(
            {
                "title": step_title,           # Description of this step
                "status": "pending",           # pending → running → done/error
                "description": "Awaiting execution",
                "substeps": [],                # Detailed logs added during execution
            }
        )

    # Launch background thread to execute the workflow
    # This allows us to return immediately without blocking the API
    # The thread will run run_agent_workflow() with the task details
    thread = threading.Thread(
        target=run_agent_workflow,
        args=(task_id, req.prompt, initial_plan_steps)
    )
    thread.start()

    # Return task_id immediately - user can poll for progress
    return {"task_id": task_id}


@app.get("/task_progress/{task_id}")
def get_task_progress(task_id: str):
    """
    Returns real-time progress information for a specific task.

    The frontend polls this endpoint every 2 seconds to update the UI
    with the current state of each step in the workflow.

    Args:
        task_id: UUID of the task to check

    Returns:
        {
            "steps": [
                {
                    "title": "Research agent: Search Tavily...",
                    "status": "done",  # or "pending", "running", "error"
                    "description": "Completed: ...",
                    "substeps": [...]  # Detailed execution logs
                },
                ...
            ]
        }

    Note: Returns empty steps array if task_id not found
    """
    return task_progress.get(task_id, {"steps": []})


@app.get("/task_status/{task_id}")
def get_task_status(task_id: str):
    """
    Retrieves the final status and results for a completed task.

    This endpoint queries the database (not the in-memory dict) to get
    the persistent task record including the final research report.

    Called by the frontend when all steps show status="done" to
    retrieve and display the final report.

    Args:
        task_id: UUID of the task to retrieve

    Returns:
        {
            "status": "done" | "running" | "error",
            "result": {
                "html_report": "Full markdown report...",
                "history": [...]  # Execution history
            }
        }

    Raises:
        HTTPException(404): If task_id does not exist in database
    """
    db = SessionLocal()
    task = db.query(Task).filter(Task.id == task_id).first()
    db.close()

    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "status": task.status,
        # Parse JSON string back to dict if result exists
        "result": json.loads(task.result) if task.result else None,
    }


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_history(history):
    """
    Formats the execution history into a readable text summary.

    This is used to provide context to agents about what previous steps
    have accomplished. Each agent can see what research was gathered,
    what drafts were written, etc.

    Args:
        history: List of [title, description, output] tuples from previous steps

    Returns:
        Formatted string with emojis and structure for readability

    Example output:
        🔹 Research agent: Search Tavily
        Gathering relevant papers...

        📝 Output:
        Found 10 papers on AI in healthcare...
    """
    return "\n\n".join(
        f"🔹 {title}\n{desc}\n\n📝 Output:\n{output}"
        for title, desc, output in history
    )


# ============================================================================
# WORKFLOW EXECUTION - The Heart of the Application
# ============================================================================

def run_agent_workflow(task_id: str, prompt: str, initial_plan_steps: list):
    """
    Executes the multi-agent research workflow in a background thread.

    This is the core orchestration function that:
    1. Loops through each planned step
    2. Executes the appropriate agent (research/writer/editor)
    3. Tracks progress and updates the UI
    4. Handles errors gracefully
    5. Saves final results to the database

    The function runs in a separate thread to avoid blocking the API.
    Users can watch progress in real-time via the /task_progress endpoint.

    Args:
        task_id: Unique identifier for this task
        prompt: Original user query
        initial_plan_steps: List of step descriptions from planner_agent

    Workflow:
        For each step:
        - Update status to "running"
        - Call executor_agent_step() which routes to the right agent
        - Collect output and add to execution_history
        - Update status to "done" with detailed substep information
        - Repeat for next step

    After all steps:
        - Extract final report (usually from last step)
        - Save to database with status="done"

    On error:
        - Mark current step as "error"
        - Save error details to database
        - Stop execution
    """
    # Get reference to the steps list in task_progress dictionary
    steps_data = task_progress[task_id]["steps"]

    # Track all outputs from each step - this builds context for future agents
    # Structure: [[step_title, description, output], ...]
    execution_history = []

    def update_step_status(index, status, description="", substep=None):
        """
        Updates the progress tracking for a specific step.

        This is called multiple times per step:
        1. When step starts (status="running")
        2. When step completes (status="done", with substep details)
        3. If step fails (status="error")

        Args:
            index: Which step to update (0-indexed)
            status: "pending" | "running" | "done" | "error"
            description: Human-readable description of current state
            substep: Optional dict with detailed execution information
        """
        if index < len(steps_data):
            steps_data[index]["status"] = status
            if description:
                steps_data[index]["description"] = description
            if substep:
                # Substeps contain detailed HTML-formatted logs
                steps_data[index]["substeps"].append(substep)
            # Add timestamp for when this update occurred
            steps_data[index]["updated_at"] = datetime.utcnow().isoformat()

    try:
        # Main execution loop - process each step sequentially
        for i, plan_step_title in enumerate(initial_plan_steps):
            # Mark this step as currently running
            update_step_status(i, "running", f"Executing: {plan_step_title}")

            # Execute this step using the appropriate agent
            # executor_agent_step determines which agent to use based on the step title
            # Returns: (step_description, agent_name, output)
            actual_step_description, agent_name, output = executor_agent_step(
                plan_step_title,      # What to do
                execution_history,    # What's been done so far
                prompt               # Original user query
            )

            # Add this step's output to the history for future steps to reference
            execution_history.append([plan_step_title, actual_step_description, output])

            # HTML escaping utilities for safe display
            def esc(s: str) -> str:
                """Escape HTML special characters to prevent XSS attacks"""
                return html.escape(s or "")

            def nl2br(s: str) -> str:
                """Convert newlines to <br> tags for HTML display"""
                return esc(s).replace("\n", "<br>")

            # Create detailed HTML-formatted log for this step
            # This is displayed in the UI when user expands the step details
            update_step_status(
                i,
                "done",
                f"Completed: {plan_step_title}",
                {
                    "title": f"Called {agent_name}",
                    # HTML content showing: user prompt, previous context, task, and output
                    "content": f"""
<div style='border:1px solid #ccc; border-radius:8px; padding:10px; margin:8px 0; background:#fff;'>
  <div style='font-weight:bold; color:#2563eb;'>📘 User Prompt</div>
  <div style='white-space:pre-wrap;'>{prompt}</div>

  <div style='font-weight:bold; color:#16a34a; margin-top:8px;'>📜 Previous Step</div>
  <pre style='white-space:pre-wrap; background:#f9fafb; padding:6px; border-radius:6px; margin:0;'>
{format_history(execution_history[-2:-1])}
  </pre>

  <div style='font-weight:bold; color:#f59e0b; margin-top:8px;'>🧹 Your next task</div>
  <div style='white-space:pre-wrap;'>{actual_step_description}</div>

  <div style='font-weight:bold; color:#10b981; margin-top:8px;'>✅ Output</div>
  <div style='white-space:pre-wrap;'>
{output}
  </div>
</div>
""".strip(),
                },
            )

        # All steps completed successfully!
        # Extract the final report (typically the output from the last step)
        final_report_markdown = (
            execution_history[-1][-1] if execution_history else "No report generated."
        )

        # Prepare result object with report and full execution history
        result = {
            "html_report": final_report_markdown,
            "history": steps_data
        }

        # Save final results to database
        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "done"
        task.result = json.dumps(result)  # Convert dict to JSON string
        task.updated_at = datetime.utcnow()
        db.commit()
        db.close()

    except Exception as e:
        # Something went wrong during execution
        print(f"Workflow error for task {task_id}: {e}")

        # Update the UI to show which step failed
        if steps_data:
            # Find the step that was running when error occurred
            error_step_index = next(
                (i for i, s in enumerate(steps_data) if s["status"] == "running"),
                len(steps_data) - 1,  # Default to last step if none running
            )
            if error_step_index >= 0:
                update_step_status(
                    error_step_index,
                    "error",
                    f"Error during execution: {e}",
                    {"title": "Error", "content": str(e)},
                )

        # Update database to reflect error state
        db = SessionLocal()
        task = db.query(Task).filter(Task.id == task_id).first()
        task.status = "error"
        task.updated_at = datetime.utcnow()
        db.commit()
        db.close()
