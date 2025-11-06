"""
agents.py - Specialized AI Agents for Research Workflow

This module contains three specialized AI agents that work together to
produce comprehensive research reports:

1. research_agent() - Information Gathering Specialist
   - Uses tool calling to search Tavily (web), arXiv (academic), Wikipedia
   - Evaluates sources for credibility and relevance
   - Synthesizes findings from multiple sources

2. writer_agent() - Academic Writing Specialist
   - Drafts comprehensive, publication-ready reports
   - Follows academic structure (abstract, intro, methods, results, etc.)
   - Ensures proper citations and references

3. editor_agent() - Quality Assurance Specialist
   - Reviews drafts for clarity and coherence
   - Improves language and structure
   - Maintains academic rigor and formatting

Each agent is implemented as a function that:
- Takes a prompt describing what to do
- Calls an LLM with specialized system instructions
- Returns formatted output for the next step in the workflow
"""

# ============================================================================
# IMPORTS
# ============================================================================

from datetime import datetime  # For timestamping research requests
from urllib import response     # Imported but not used (can be removed)
from aisuite import Client      # AI client for LLM calls

# Import research tool definitions
from src.research_tools import (
    arxiv_search_tool,      # Searches academic papers on arXiv
    tavily_search_tool,     # Searches web using Tavily API
    wikipedia_search_tool,  # Searches Wikipedia for background info
)

# ============================================================================
# AI CLIENT INITIALIZATION
# ============================================================================

# Create shared client instance for all agent LLM calls
client = Client()


# ============================================================================
# RESEARCH AGENT - Information Gathering with Tool Use
# ============================================================================

def research_agent(
    prompt: str,
    model: str = "openai:gpt-4.1-mini",
    return_messages: bool = False
):
    """
    Gathers comprehensive research information using multiple search tools.

    This agent is the information-gathering workhorse of the system. It:
    1. Receives a research task (e.g., "Search Tavily for papers on AI")
    2. Decides which tools to use (Tavily, arXiv, Wikipedia)
    3. Executes multiple searches strategically
    4. Synthesizes findings into a structured summary

    The agent uses "tool calling" - the LLM can invoke Python functions
    (search tools) to gather real-time information, then incorporates that
    data into its response.

    Args:
        prompt: Research task description, including context from previous steps
                Example: "Use Tavily to find recent papers on transformers in NLP"
        model: LLM to use (default: gpt-4.1-mini for speed and cost)
        return_messages: If True, also return message history (unused currently)

    Returns:
        Tuple of (content, messages):
        - content: Formatted research findings with source attribution and
                   HTML section listing which tools were used
        - messages: Message history from the LLM conversation

    How Tool Calling Works:
        1. Agent is given tool definitions (arxiv_search_tool, etc.)
        2. LLM decides which tools to call and with what arguments
        3. Tools execute and return results (paper titles, URLs, etc.)
        4. LLM incorporates results into final summary
        5. Can make up to 5 tool calls (max_turns=5)

    Example:
        Input:  prompt = "Search for transformer architecture papers"
        Agent thinking: "I should use arxiv_search_tool for academic papers"
        Tool call: arxiv_search_tool(query="transformer architecture")
        Tool result: [{"title": "Attention Is All You Need", ...}, ...]
        Agent output: "Found 5 relevant papers on transformer architecture..."
    """

    # Print header for logging/debugging
    print("==================================")
    print("🔍 Research Agent")
    print("==================================")

    # ========================================================================
    # CONSTRUCT SYSTEM PROMPT - Defines Agent Behavior
    # ========================================================================

    # This prompt gives the agent its "personality" and instructions
    full_prompt = f"""
You are an advanced research assistant with expertise in information retrieval and academic research methodology. Your mission is to gather comprehensive, accurate, and relevant information on any topic requested by the user.

## AVAILABLE RESEARCH TOOLS:

1. **`tavily_search_tool`**: General web search engine
   - USE FOR: Recent news, current events, blogs, websites, industry reports, and non-academic sources
   - BEST FOR: Up-to-date information, diverse perspectives, practical applications, and general knowledge

2. **`arxiv_search_tool`**: Academic publication database
   - USE FOR: Peer-reviewed research papers, technical reports, and scholarly articles
   - LIMITED TO THESE DOMAINS ONLY:
     * Computer Science
     * Mathematics
     * Physics
     * Statistics
     * Quantitative Biology
     * Quantitative Finance
     * Electrical Engineering and Systems Science
     * Economics
   - BEST FOR: Scientific evidence, theoretical frameworks, and technical details in supported fields

3. **`wikipedia_search_tool`**: Encyclopedia resource
   - USE FOR: Background information, definitions, overviews, historical context
   - BEST FOR: Establishing foundational knowledge and understanding basic concepts

## RESEARCH METHODOLOGY:

1. **Analyze Request**: Identify the core research questions and knowledge domains
2. **Plan Search Strategy**: Determine which tools are most appropriate for the topic
3. **Execute Searches**: Use the selected tools with effective keywords and queries
4. **Evaluate Sources**: Prioritize credibility, relevance, recency, and diversity
5. **Synthesize Findings**: Organize information logically with clear source attribution
6. **Document Search Process**: Note which tools were used and why

## TOOL SELECTION GUIDELINES:

- For scientific/academic questions in supported domains → Use `arxiv_search_tool`
- For recent developments, news, or practical information → Use `tavily_search_tool`
- For fundamental concepts or historical context → Use `wikipedia_search_tool`
- For comprehensive research → Use multiple tools strategically
- NEVER use `arxiv_search_tool` for domains outside its supported list
- ALWAYS verify information across multiple sources when possible

## OUTPUT FORMAT:

Present your research findings in a structured format that includes:
1. **Summary of Research Approach**: Tools used and search strategy
2. **Key Findings**: Organized by subtopic or source
3. **Source Details**: Include URLs, titles, authors, and publication dates
4. **Limitations**: Note any gaps in available information

Today is {datetime.now().strftime("%Y-%m-%d")}.

USER RESEARCH REQUEST:
{prompt}
""".strip()

    # Prepare messages for LLM API call
    messages = [{"role": "user", "content": full_prompt}]

    # Define which tools the agent can use
    tools = [arxiv_search_tool, tavily_search_tool, wikipedia_search_tool]

    try:
        # ====================================================================
        # CALL LLM WITH TOOL USE ENABLED
        # ====================================================================

        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,                 # Enable tool calling
            tool_choice="auto",          # Let LLM decide when to use tools
            max_turns=5,                 # Allow up to 5 tool invocations
            temperature=0.0,             # Deterministic responses for research
        )

        # Extract the main content from the response
        content = resp.choices[0].message.content or ""

        # ====================================================================
        # EXTRACT AND DISPLAY TOOL USAGE
        # ====================================================================

        # Track which tools were called during the research process
        calls = []

        # Method A: Extract from intermediate_responses
        # Some LLM APIs provide intermediate steps showing tool calls
        for ir in getattr(resp, "intermediate_responses", []) or []:
            try:
                tcs = ir.choices[0].message.tool_calls or []
                for tc in tcs:
                    calls.append((tc.function.name, tc.function.arguments))
            except Exception:
                pass  # Skip if this response doesn't have tool calls

        # Method B: Extract from intermediate_messages
        # Alternative location where tool calls might be stored
        for msg in getattr(resp.choices[0].message, "intermediate_messages", []) or []:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    calls.append((tc.function.name, tc.function.arguments))

        # Deduplicate tool calls while preserving order
        # (Sometimes the same call appears in multiple places)
        seen = set()
        dedup_calls = []
        for name, args in calls:
            key = (name, args)
            if key not in seen:
                seen.add(key)
                dedup_calls.append((name, args))

        # Format tool calls for display
        tool_lines = []
        for name, args in dedup_calls:
            arg_text = str(args)
            try:
                import json as _json

                # Try to parse arguments as JSON for prettier display
                parsed = _json.loads(args) if isinstance(args, str) else args
                if isinstance(parsed, dict):
                    # Format as key=value pairs
                    kv = ", ".join(f"{k}={repr(v)}" for k, v in parsed.items())
                    arg_text = kv
            except Exception:
                # Keep raw string if not JSON
                pass
            tool_lines.append(f"- {name}({arg_text})")

        # Add HTML section showing which tools were used
        if tool_lines:
            tools_html = (
                "<h2 style='font-size:1.5em; color:#2563eb;'>📎 Tools used</h2>"
            )
            tools_html += (
                "<ul>" + "".join(f"<li>{line}</li>" for line in tool_lines) + "</ul>"
            )
            content += "\n\n" + tools_html

        # Log output for debugging
        print("✅ Output:\n", content)
        return content, messages

    except Exception as e:
        # Handle errors gracefully - return error message instead of crashing
        print("❌ Error:", e)
        return f"[Model Error: {str(e)}]", messages


# ============================================================================
# WRITER AGENT - Academic Report Generation
# ============================================================================

def writer_agent(
    prompt: str,
    model: str = "openai:gpt-4.1-mini",
    min_words_total: int = 2400,        # Unused but kept for compatibility
    min_words_per_section: int = 400,   # Unused but kept for compatibility
    max_tokens: int = 15000,            # Maximum response length
    retries: int = 1,                   # Unused but kept for compatibility
):
    """
    Creates comprehensive, publication-ready academic reports.

    This agent specializes in scholarly writing. Given research findings
    from the research agent, it produces well-structured reports following
    academic conventions.

    The writer agent:
    - Follows a strict structure (Abstract, Intro, Methods, Results, etc.)
    - Uses formal academic language
    - Includes proper citations and references
    - Synthesizes information into coherent narrative (not just summarizing)

    Args:
        prompt: Writing task including research findings from previous steps
                The prompt contains all gathered research that should be
                incorporated into the report
        model: LLM to use (default: gpt-4.1-mini)
        min_words_total: Target minimum word count (currently not enforced)
        min_words_per_section: Minimum words per section (currently not enforced)
        max_tokens: Maximum tokens in response (15000 ≈ 11,250 words)
        retries: Number of retries if output too short (currently not used)

    Returns:
        Tuple of (content, messages):
        - content: Complete academic report in Markdown format
        - messages: Message history (system + user prompts)

    Report Structure:
        1. Title - Clear and descriptive
        2. Abstract - 100-150 word summary
        3. Introduction - Context and research questions
        4. Background/Literature Review - Previous work
        5. Methodology - Research approach (if applicable)
        6. Key Findings/Results - Main discoveries
        7. Discussion - Analysis and implications
        8. Conclusion - Summary and future directions
        9. References - Complete bibliography with links

    Example:
        Input:  prompt = "Draft report on AI in healthcare\n\nFindings: ..."
        Output: Complete markdown report with title, abstract, sections, citations
    """

    # Print header for logging
    print("==================================")
    print("✍️ Writer Agent")
    print("==================================")

    # ========================================================================
    # SYSTEM PROMPT - Defines Academic Writing Standards
    # ========================================================================

    # This detailed prompt teaches the LLM how to write academic reports
    system_message = """
You are an expert academic writer with a PhD-level understanding of scholarly communication. Your task is to synthesize research materials into a comprehensive, well-structured academic report.

## REPORT REQUIREMENTS:
- Produce a COMPLETE, POLISHED, and PUBLICATION-READY academic report in Markdown format
- Create original content that thoroughly analyzes the provided research materials
- DO NOT merely summarize the sources; develop a cohesive narrative with critical analysis
- Length should be appropriate to thoroughly cover the topic (typically 1500-3000 words)

## MANDATORY STRUCTURE:
1. **Title**: Clear, concise, and descriptive of the content
2. **Abstract**: Brief summary (100-150 words) of the report's purpose, methods, and key findings
3. **Introduction**: Present the topic, research question/problem, significance, and outline of the report
4. **Background/Literature Review**: Contextualize the topic within existing scholarship
5. **Methodology**: If applicable, describe research methods, data collection, and analytical approaches
6. **Key Findings/Results**: Present the primary outcomes and evidence
7. **Discussion**: Interpret findings, address implications, limitations, and connections to broader field
8. **Conclusion**: Synthesize main points and suggest directions for future research
9. **References**: Complete list of all cited works

## ACADEMIC WRITING GUIDELINES:
- Maintain formal, precise, and objective language throughout
- Use discipline-appropriate terminology and concepts
- Support all claims with evidence and reasoning
- Develop logical flow between ideas, paragraphs, and sections
- Include relevant examples, case studies, data, or equations to strengthen arguments
- Address potential counterarguments and limitations

## CITATION AND REFERENCE RULES:
- Use numeric inline citations [1], [2], etc. for all borrowed ideas and information
- Every claim based on external sources MUST have a citation
- Each inline citation must correspond to a complete entry in the References section
- Every reference listed must be cited at least once in the text
- Preserve ALL original URLs, DOIs, and bibliographic information from source materials
- Format references consistently according to academic standards

## FORMATTING GUIDELINES:
- Use Markdown syntax for all formatting (headings, emphasis, lists, etc.)
- Include appropriate section headings and subheadings to organize content
- Format any equations, tables, or figures according to academic conventions
- Use bullet points or numbered lists when appropriate for clarity
- Use html syntax to handle all links with target="_blank", so user can always open link in new tab on both html and markdown format

Output the complete report in Markdown format only. Do not include meta-commentary about the writing process.

INTERNAL CHECKLIST (DO NOT INCLUDE IN OUTPUT):
- [ ] Incorporated all provided research materials
- [ ] Developed original analysis beyond mere summarization
- [ ] Included all mandatory sections with appropriate content
- [ ] Used proper inline citations for all borrowed content
- [ ] Created complete References section with all cited sources
- [ ] Maintained academic tone and language throughout
- [ ] Ensured logical flow and coherent structure
- [ ] Preserved all source URLs and bibliographic information
""".strip()

    # Prepare messages with system prompt + user request
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================

    def _call(messages_):
        """Helper to make LLM API call."""
        resp = client.chat.completions.create(
            model=model,
            messages=messages_,
            temperature=0,           # Deterministic for consistent quality
            max_tokens=max_tokens,   # Allow long reports
        )
        return resp.choices[0].message.content or ""

    def _word_count(md_text: str) -> int:
        """Count words in markdown text (currently unused)."""
        import re
        words = re.findall(r"\b\w+\b", md_text)
        return len(words)

    # ========================================================================
    # GENERATE REPORT
    # ========================================================================

    # Call LLM to generate the report
    content = _call(messages)

    # Log output
    print("✅ Output:\n", content)
    return content, messages


# ============================================================================
# EDITOR AGENT - Quality Assurance and Polish
# ============================================================================

def editor_agent(
    prompt: str,
    model: str = "openai:gpt-4.1-mini",
    target_min_words: int = 2400,  # Unused but kept for compatibility
):
    """
    Reviews and improves academic writing quality.

    This agent acts as a professional editor. It receives a draft report
    (usually from writer_agent) and:
    - Improves clarity and readability
    - Ensures logical flow and coherence
    - Standardizes terminology
    - Enhances academic tone
    - Preserves all citations and references

    The editor does NOT add new research - it only improves the presentation
    of existing content.

    Args:
        prompt: Editing task including the draft to be improved
                Contains both the draft text and any specific editing instructions
        model: LLM to use (default: gpt-4.1-mini)
        target_min_words: Target word count (currently not enforced)

    Returns:
        Tuple of (content, messages):
        - content: Revised and polished report in Markdown format
        - messages: Message history (system + user prompts)

    Editing Focus Areas:
        - Structure and organization
        - Argument clarity and strength
        - Language precision and conciseness
        - Transition quality between sections
        - Citation completeness and formatting
        - Academic tone consistency

    Example:
        Input:  prompt = "Review this draft: [draft text]"
        Output: Improved version with better flow, clarity, and polish
    """

    # Print header for logging
    print("==================================")
    print("🧠 Editor Agent")
    print("==================================")

    # ========================================================================
    # SYSTEM PROMPT - Defines Editing Standards
    # ========================================================================

    system_message = """
You are a professional academic editor with expertise in improving scholarly writing across disciplines. Your task is to refine and elevate the quality of the academic text provided.

## Your Editing Process:
1. Analyze the overall structure, argument flow, and coherence of the text
2. Ensure logical progression of ideas with clear topic sentences and transitions between paragraphs
3. Improve clarity, precision, and conciseness of language while maintaining academic tone
4. Verify technical accuracy (to the extent possible based on context)
5. Enhance readability through appropriate formatting and organization

## Specific Elements to Address:
- Strengthen thesis statements and main arguments
- Clarify complex concepts with additional explanations or examples where needed
- Add relevant equations, diagrams, or illustrations (described in markdown) when they would enhance understanding
- Ensure proper integration of evidence and maintain academic rigor
- Standardize terminology and eliminate redundancies
- Improve sentence variety and paragraph structure
- Preserve all citations [1], [2], etc., and maintain the integrity of the References section

## Formatting Guidelines:
- Use markdown formatting consistently for headings, emphasis, lists, etc.
- Structure content with appropriate section headings and subheadings
- Format equations, tables, and figures according to academic standards

Return only the revised, polished text in Markdown format without explanatory comments about your edits.
""".strip()

    # Prepare messages
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt},
    ]

    # ========================================================================
    # GENERATE EDITED VERSION
    # ========================================================================

    # Call LLM to edit the content
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0  # Deterministic editing
    )

    # Extract edited content
    content = response.choices[0].message.content

    # Log output
    print("✅ Output:\n", content)
    return content, messages
