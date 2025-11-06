"""
planning_agent.py - Research Workflow Planning and Execution Routing

This module provides the planning and coordination layer for the multi-agent
research system. It contains two key components:

1. planner_agent() - Uses an AI model to create a structured research plan
2. executor_agent_step() - Routes each step to the appropriate specialized agent

The planning agent ensures a consistent research methodology:
- Step 1: Always starts with broad web search (Tavily)
- Step 2: Follows up with targeted arXiv search
- Step 3-6: Synthesis, drafting, and editing
- Step 7: Final report generation with citations

This creates a reliable, repeatable research workflow.
"""

# ============================================================================
# IMPORTS
# ============================================================================

import json  # For parsing JSON responses from AI models
import re    # For regex-based string cleaning and extraction
from typing import List  # For type hints indicating list of strings
from datetime import datetime  # For timestamping (used in context)
import ast   # For safely parsing Python literal expressions

# AI client for making LLM calls
from aisuite import Client

# Import specialized agent functions
from src.agents import (
    research_agent,  # Gathers information using search tools
    writer_agent,    # Drafts comprehensive reports
    editor_agent,    # Reviews and polishes content
)

# ============================================================================
# AI CLIENT INITIALIZATION
# ============================================================================

# Create a single shared client instance for all LLM calls
# This client handles API authentication and request management
client = Client()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def clean_json_block(raw: str) -> str:
    """
    Removes markdown code fence formatting from JSON strings.

    AI models sometimes wrap JSON output in markdown code blocks like:
    ```json
    ["item1", "item2"]
    ```

    This function strips those markers to get clean JSON.

    Args:
        raw: String that may contain markdown code fences

    Returns:
        Cleaned string with code fences removed

    Example:
        Input:  "```json\n[\"step1\", \"step2\"]\n```"
        Output: "[\"step1\", \"step2\"]"
    """
    raw = raw.strip()

    # Remove opening code fence (e.g., ```json or ```python)
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z]*\n?", "", raw)
        # Remove closing code fence
        raw = re.sub(r"\n?```$", "", raw)

    # Clean up any remaining backticks and whitespace
    return raw.strip("` \n")


# ============================================================================
# PLANNING AGENT - Creates Research Workflow
# ============================================================================

def planner_agent(topic: str, model: str = "openai:o4-mini") -> List[str]:
    """
    Creates a structured, step-by-step research plan for a given topic.

    This is the first function called when a user submits a research query.
    It uses an AI model (o4-mini by default) to generate a list of actionable
    steps that will be executed by specialized agents.

    The planner enforces a specific methodology:
    1. Broad web search (Tavily) to find authoritative sources
    2. Targeted arXiv search for academic papers
    3. Synthesis and ranking of findings
    4. Drafting and editing
    5. Final report with citations

    Args:
        topic: The research question or topic from the user
               Example: "AI applications in drug discovery"
        model: AI model to use for planning (default: openai:o4-mini)
               Can be changed to other models like "openai:gpt-4o"

    Returns:
        List of 3-7 step descriptions (strings), each assigned to an agent
        Example: [
            "Research agent: Use Tavily to perform a broad web search...",
            "Research agent: For each collected item, search on arXiv...",
            "Writer agent: Draft a structured outline...",
            ...
        ]

    How it works:
        1. Sends a detailed prompt to the AI model describing available agents
        2. Asks the model to create a research plan as a Python list
        3. Parses the response (handles JSON, Python literals, code fences)
        4. Enforces required first/last steps for consistency
        5. Caps the plan at 7 steps maximum

    Note: The function includes fallback logic to ensure valid plans even
          if the AI model returns unexpected formats.
    """

    # Construct the prompt for the AI planner
    # This prompt defines the "rules" for creating a good research plan
    prompt = f"""
You are a planning agent responsible for organizing a research workflow using multiple intelligent agents.

🧠 Available agents:
- Research agent: MUST begin with a broad **web search using Tavily** to identify only **relevant** and **authoritative** items (e.g., high-impact venues, seminal works, surveys, or recent comprehensive sources). The output of this step MUST capture for each candidate: title, authors, year, venue/source, URL, and (if available) DOI.
- Research agent: AFTER the Tavily step, perform a **targeted arXiv search** ONLY for the candidates discovered in the web step (match by title/author/DOI). If an arXiv preprint/version exists, record its arXiv URL and version info. Do NOT run a generic arXiv search detached from the Tavily results.
- Writer agent: drafts based on research findings.
- Editor agent: reviews, reflects on, and improves drafts.

🎯 Produce a clear step-by-step research plan **as a valid Python list of strings** (no markdown, no explanations).
Each step must be atomic, actionable, and assigned to one of the agents.
Maximum of 7 steps.

🚫 DO NOT include steps like "create CSV", "set up repo", "install packages".
✅ Focus on meaningful research tasks (search, extract, rank, draft, revise).
✅ The FIRST step MUST be exactly:
"Research agent: Use Tavily to perform a broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
✅ The SECOND step MUST be exactly:
"Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."

🔚 The FINAL step MUST instruct the writer agent to generate a comprehensive Markdown report that:
- Uses all findings and outputs from previous steps
- Includes inline citations (e.g., [1], (Wikipedia/arXiv))
- Includes a References section with clickable links for all citations
- Preserves earlier sources
- Is detailed and self-contained

Topic: "{topic}"
"""

    # Call the AI model to generate the plan
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1,  # Some creativity allowed in planning
    )

    # Extract the text response
    raw = response.choices[0].message.content.strip()

    # ========================================================================
    # ROBUST PARSING - Handle Multiple Response Formats
    # ========================================================================

    def _coerce_to_list(s: str) -> List[str]:
        """
        Attempts to parse a string into a list of strings using multiple methods.

        The AI model might return the plan in various formats:
        - Valid JSON: ["step1", "step2"]
        - Python list: ['step1', 'step2']
        - Markdown code fence: ```["step1", "step2"]```

        This function tries each parsing method until one succeeds.

        Args:
            s: String potentially containing a list

        Returns:
            List of strings (max 7 items), or empty list if parsing fails
        """

        # Attempt 1: Try parsing as strict JSON
        try:
            obj = json.loads(s)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj[:7]  # Limit to 7 steps
        except json.JSONDecodeError:
            pass  # Not valid JSON, try next method

        # Attempt 2: Try parsing as Python literal (handles single quotes)
        try:
            obj = ast.literal_eval(s)
            if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                return obj[:7]
        except Exception:
            pass  # Not a valid Python literal

        # Attempt 3: Try extracting from code fence
        if s.startswith("```") and s.endswith("```"):
            # Remove backticks and try parsing again
            inner = s.strip("`")
            try:
                obj = ast.literal_eval(inner)
                if isinstance(obj, list) and all(isinstance(x, str) for x in obj):
                    return obj[:7]
            except Exception:
                pass

        # All parsing attempts failed
        return []

    # Parse the AI response into a list of steps
    steps = _coerce_to_list(raw)

    # ========================================================================
    # ENFORCE RESEARCH METHODOLOGY CONTRACT
    # ========================================================================

    # Define required steps to ensure consistent research quality
    required_first = "Research agent: Use Tavily to perform a broad web search and collect top relevant items (title, authors, year, venue/source, URL, DOI if available)."
    required_second = "Research agent: For each collected item, search on arXiv to find matching preprints/versions and record arXiv URLs (if they exist)."
    final_required = "Writer agent: Generate the final comprehensive Markdown report with inline citations and a complete References section with clickable links."

    def _ensure_contract(steps_list: List[str]) -> List[str]:
        """
        Ensures the research plan meets quality standards.

        This function enforces that:
        1. First step is always a broad Tavily search
        2. Second step is always a targeted arXiv search
        3. Final step always generates a comprehensive report
        4. Total steps don't exceed 7

        If the AI model didn't follow these rules, this function fixes it.

        Args:
            steps_list: Raw list of steps from the AI model

        Returns:
            Validated and corrected list of steps
        """

        # If AI returned empty or invalid response, use default plan
        if not steps_list:
            return [
                required_first,
                required_second,
                "Research agent: Synthesize and rank findings by relevance, recency, and authority; deduplicate by title/DOI.",
                "Writer agent: Draft a structured outline based on the ranked evidence.",
                "Editor agent: Review for coherence, coverage, and citation completeness; request fixes.",
                final_required,
            ]

        # Filter out any non-string items
        steps_list = [s for s in steps_list if isinstance(s, str)]

        # Ensure first step is the required Tavily search
        if not steps_list or steps_list[0] != required_first:
            steps_list = [required_first] + steps_list

        # Ensure second step is the required arXiv search
        if len(steps_list) < 2 or steps_list[1] != required_second:
            # Remove any generic arXiv steps not tied to Tavily results
            steps_list = (
                [steps_list[0]]
                + [required_second]
                + [
                    s
                    for s in steps_list[1:]
                    if "arXiv" not in s or "For each collected item" in s
                ]
            )

        # Ensure final step generates comprehensive report
        if final_required not in steps_list:
            steps_list.append(final_required)

        # Cap at 7 steps maximum
        return steps_list[:7]

    # Apply contract enforcement
    steps = _ensure_contract(steps)

    return steps


# ============================================================================
# EXECUTOR AGENT - Routes Steps to Specialized Agents
# ============================================================================

def executor_agent_step(step_title: str, history: list, prompt: str):
    """
    Executes a single step by routing it to the appropriate specialized agent.

    This function acts as a dispatcher/router. Based on keywords in the step
    description, it decides which specialized agent to call:
    - "research" → research_agent() (uses Tavily, arXiv, Wikipedia tools)
    - "draft" or "write" → writer_agent() (creates comprehensive reports)
    - "edit" or "revise" → editor_agent() (reviews and polishes content)

    The function also builds rich context by formatting the execution history
    so each agent knows what previous agents have accomplished.

    Args:
        step_title: Description of this step (e.g., "Research agent: Search Tavily...")
        history: List of [title, description, output] from all previous steps
                 This provides context so agents can build on prior work
        prompt: The original user research query

    Returns:
        Tuple of (step_title, agent_name, output):
        - step_title: The step description (passed through)
        - agent_name: Which agent was used ("research_agent", "writer_agent", etc.)
        - output: The agent's response/output (markdown text, research findings, etc.)

    Example:
        Input:  step_title = "Research agent: Use Tavily to search..."
                history = []
                prompt = "AI in healthcare"
        Output: ("Research agent: Use Tavily...", "research_agent", "Found 10 papers...")

    How context is built:
        The function categorizes previous steps by agent type and formats them
        with emojis for readability:
        - 🔍 Research findings
        - ✍️ Draft content
        - 🧠 Editor feedback

        This formatted history is prepended to the step description so the
        agent has full context of what's been done.
    """

    # ========================================================================
    # BUILD ENRICHED CONTEXT
    # ========================================================================

    # Start with the user's original query
    context = f"📘 User Prompt:\n{prompt}\n\n📜 History so far:\n"

    # Format each previous step with appropriate emoji and label
    for i, (desc, agent, output) in enumerate(history):
        # Categorize by agent type or keywords in description
        if "draft" in desc.lower() or agent == "writer_agent":
            context += f"\n✍️ Draft (Step {i + 1}):\n{output.strip()}\n"
        elif "feedback" in desc.lower() or agent == "editor_agent":
            context += f"\n🧠 Feedback (Step {i + 1}):\n{output.strip()}\n"
        elif "research" in desc.lower() or agent == "research_agent":
            context += f"\n🔍 Research (Step {i + 1}):\n{output.strip()}\n"
        else:
            # Unknown agent type - use generic label
            context += f"\n🧩 Other (Step {i + 1}) by {agent}:\n{output.strip()}\n"

    # Combine context with current task
    enriched_task = f"""{context}

🧩 Your next task:
{step_title}
"""

    # ========================================================================
    # ROUTE TO APPROPRIATE AGENT
    # ========================================================================

    # Convert step title to lowercase for case-insensitive matching
    step_lower = step_title.lower()

    # Route based on keywords in the step description
    if "research" in step_lower:
        # Research agent: Gathers information using search tools
        content, _ = research_agent(prompt=enriched_task)
        print("🔍 Research Agent Output:", content)  # Log for debugging
        return step_title, "research_agent", content

    elif "draft" in step_lower or "write" in step_lower:
        # Writer agent: Creates comprehensive reports
        content, _ = writer_agent(prompt=enriched_task)
        return step_title, "writer_agent", content

    elif "revise" in step_lower or "edit" in step_lower or "feedback" in step_lower:
        # Editor agent: Reviews and improves content
        content, _ = editor_agent(prompt=enriched_task)
        return step_title, "editor_agent", content

    else:
        # Unknown step type - this shouldn't happen if planner is working correctly
        raise ValueError(f"Unknown step type: {step_title}")
