"""
research_tools.py - Search and Information Retrieval Tools

This module provides three main research tools that AI agents can call
to gather information from different sources:

1. arXiv Search Tool - Academic paper database
   - Searches arXiv API for research papers
   - Downloads PDFs and extracts text content
   - Focuses on CS, Math, Physics, Stats, and related fields

2. Tavily Search Tool - General web search
   - Uses Tavily API for broad web searches
   - Retrieves recent news, blogs, reports, and websites
   - Good for current events and diverse perspectives

3. Wikipedia Search Tool - Encyclopedia lookups
   - Searches Wikipedia for background information
   - Provides definitions, context, and overviews
   - Useful for establishing foundational knowledge

Each tool is designed to be called by LLMs via "function calling" -
the AI model can invoke these functions with specific queries and
incorporate the results into its research findings.
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Type hints
from typing import List, Dict, Optional

# Standard library
import os       # For environment variables and file operations
import re       # For text cleaning and regex operations
import time     # For rate limiting delays
from io import BytesIO  # For in-memory PDF handling

# HTTP requests with retry logic
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# XML parsing for arXiv API responses
import xml.etree.ElementTree as ET

# PDF text extraction libraries
from pdfminer.high_level import extract_text_to_fp  # Backup PDF parser

# Environment and API clients
from dotenv import load_dotenv      # Load .env file
from tavily import TavilyClient     # Tavily search API client
import wikipedia                     # Wikipedia API wrapper

# Load environment variables (TAVILY_API_KEY, etc.)
load_dotenv()


# ============================================================================
# HTTP SESSION WITH RETRY LOGIC
# ============================================================================

def _build_session(
    user_agent: str = "LF-ADP-Agent/1.0 (mailto:your.email@example.com)",
) -> requests.Session:
    """
    Creates an HTTP session with automatic retry logic and proper headers.

    When making requests to external APIs (arXiv, PDF downloads), network
    issues or rate limiting can cause failures. This session automatically
    retries failed requests with exponential backoff.

    Args:
        user_agent: User-Agent string identifying this application
                    APIs use this to track usage and enforce rate limits

    Returns:
        Configured requests.Session object with:
        - Retry logic (up to 5 retries)
        - Backoff delays (0.6s, 1.2s, 2.4s, ...)
        - Connection pooling for performance

    Retry Strategy:
        - Retries on: 429 (Too Many Requests), 500+ (Server Errors)
        - Backoff factor: 0.6 seconds
        - Max retries: 5 attempts for connection, read, and total
        - Only retries GET and HEAD requests (not POST)
    """
    s = requests.Session()

    # Set standard headers
    s.headers.update(
        {
            "User-Agent": user_agent,
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
    )

    # Configure retry strategy
    retry = Retry(
        total=5,                                  # Maximum total retries
        connect=5,                                # Max connection retries
        read=5,                                   # Max read retries
        backoff_factor=0.6,                       # Wait 0.6s, 1.2s, 2.4s, ...
        status_forcelist=(429, 500, 502, 503, 504),  # Retry these HTTP codes
        allowed_methods=frozenset(["GET", "HEAD"]),  # Only retry safe methods
        raise_on_redirect=False,                  # Don't fail on redirects
        raise_on_status=False,                    # Let caller handle errors
    )

    # Attach retry logic to HTTPS and HTTP
    adapter = HTTPAdapter(
        max_retries=retry,
        pool_connections=10,   # Keep 10 connections alive
        pool_maxsize=20        # Up to 20 connections total
    )
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    return s


# Create a shared session instance used by all tools
session = _build_session()


# ============================================================================
# PDF UTILITY FUNCTIONS
# ============================================================================

def ensure_pdf_url(abs_or_pdf_url: str) -> str:
    """
    Converts an arXiv abstract URL to a PDF download URL.

    arXiv has two URL formats:
    - Abstract page: https://arxiv.org/abs/2106.09685
    - PDF download:  https://arxiv.org/pdf/2106.09685.pdf

    This function ensures we have the PDF URL for downloading.

    Args:
        abs_or_pdf_url: Either abstract or PDF URL from arXiv

    Returns:
        PDF download URL with HTTPS and .pdf extension

    Example:
        Input:  "http://arxiv.org/abs/2106.09685"
        Output: "https://arxiv.org/pdf/2106.09685.pdf"
    """
    # Upgrade to HTTPS
    url = abs_or_pdf_url.strip().replace("http://", "https://")

    # If already a PDF URL, return as-is
    if "/pdf/" in url and url.endswith(".pdf"):
        return url

    # Convert /abs/ to /pdf/
    url = url.replace("/abs/", "/pdf/")

    # Ensure .pdf extension
    if not url.endswith(".pdf"):
        url += ".pdf"

    return url


def _safe_filename(name: str) -> str:
    """
    Sanitizes a string to be a safe filename.

    Removes special characters that could cause file system issues
    or security problems (path traversal, etc.).

    Args:
        name: Proposed filename

    Returns:
        Sanitized filename safe for all operating systems

    Example:
        Input:  "My Paper: A Study (2023)!.pdf"
        Output: "My_Paper__A_Study__2023__.pdf"
    """
    # Replace unsafe characters with underscores
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)

    # Ensure .pdf extension
    if not name.lower().endswith(".pdf"):
        name += ".pdf"

    return name


def clean_text(s: str) -> str:
    """
    Cleans extracted PDF text for better readability.

    PDF extraction often includes artifacts like:
    - Hyphenated line breaks ("transfor-\\nmers")
    - Inconsistent line endings (\\r\\n vs \\n)
    - Multiple spaces
    - Excessive blank lines

    This function normalizes the text.

    Args:
        s: Raw text extracted from PDF

    Returns:
        Cleaned text with normalized whitespace and line breaks

    Example:
        Input:  "transfor-\\nmers  use\\r\\nattention\\n\\n\\n\\nmechanisms"
        Output: "transformers use\\nattention\\n\\nmechanisms"
    """
    # Remove hyphenated line breaks
    s = re.sub(r"-\n", "", s)

    # Normalize line endings to \\n
    s = re.sub(r"\r\n|\r", "\n", s)

    # Collapse multiple spaces/tabs to single space
    s = re.sub(r"[ \t]+", " ", s)

    # No more than one blank line in a row
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip()


def fetch_pdf_bytes(pdf_url: str, timeout: int = 90) -> bytes:
    """
    Downloads a PDF file from a URL into memory.

    Args:
        pdf_url: URL to PDF file
        timeout: Maximum seconds to wait for download

    Returns:
        Raw PDF bytes

    Raises:
        requests.HTTPError: If download fails (404, 500, etc.)
        requests.Timeout: If download takes longer than timeout

    Note:
        Uses the shared session with retry logic, so transient
        failures are automatically retried.
    """
    r = session.get(pdf_url, timeout=timeout, allow_redirects=True)
    r.raise_for_status()  # Raise exception if status code indicates error
    return r.content


def pdf_bytes_to_text(pdf_bytes: bytes, max_pages: Optional[int] = None) -> str:
    """
    Extracts text content from PDF bytes using multiple fallback methods.

    Try 1: PyMuPDF (fitz) - Fast and accurate
    Try 2: pdfminer.six - Slower but handles complex layouts

    This two-tier approach ensures we can extract text from most PDFs.

    Args:
        pdf_bytes: Raw PDF data
        max_pages: Optional limit on pages to extract (for speed)
                   None = extract all pages

    Returns:
        Extracted text as a single string

    Raises:
        RuntimeError: If all extraction methods fail

    Example:
        pdf_bytes = fetch_pdf_bytes("https://arxiv.org/pdf/2106.09685.pdf")
        text = pdf_bytes_to_text(pdf_bytes, max_pages=10)
    """
    # Attempt 1: Try PyMuPDF (fastest and most accurate)
    try:
        import fitz  # PyMuPDF library

        out = []
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            n = len(doc)
            limit = n if max_pages is None else min(max_pages, n)
            for i in range(limit):
                out.append(doc.load_page(i).get_text("text"))
        return "\n".join(out)
    except Exception:
        pass  # Fall through to backup method

    # Attempt 2: Try pdfminer.six (fallback)
    try:
        buf_in = BytesIO(pdf_bytes)
        buf_out = BytesIO()
        extract_text_to_fp(buf_in, buf_out)
        return buf_out.getvalue().decode("utf-8", errors="ignore")
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {e}")


def maybe_save_pdf(pdf_bytes: bytes, dest_dir: str, filename: str) -> str:
    """
    Optionally saves PDF bytes to disk (currently unused but available).

    Args:
        pdf_bytes: Raw PDF data
        dest_dir: Directory to save in (created if doesn't exist)
        filename: Desired filename (will be sanitized)

    Returns:
        Full path to saved file

    Example:
        path = maybe_save_pdf(pdf_bytes, "/tmp/papers", "attention.pdf")
        # Returns: "/tmp/papers/attention.pdf"
    """
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, _safe_filename(filename))
    with open(path, "wb") as f:
        f.write(pdf_bytes)
    return path


# ============================================================================
# ARXIV SEARCH TOOL - Academic Paper Search with PDF Extraction
# ============================================================================

def arxiv_search_tool(
    query: str,
    max_results: int = 3,
) -> List[Dict]:
    """
    Searches arXiv for academic papers and extracts their full text.

    This tool:
    1. Queries arXiv API with the search term
    2. Retrieves metadata (title, authors, abstract, URL)
    3. Downloads the PDF for each result
    4. Extracts text from the first 6 pages
    5. Returns enriched results with paper content

    The extracted text allows the LLM to analyze actual paper content,
    not just abstracts, leading to more informed research summaries.

    Args:
        query: Search keywords (e.g., "transformer architecture NLP")
        max_results: Maximum papers to return (default 3, more = slower)

    Returns:
        List of dictionaries, each containing:
        - title: Paper title
        - authors: List of author names
        - published: Publication date (YYYY-MM-DD)
        - url: Link to arXiv abstract page
        - summary: First 5000 characters of extracted PDF text
                   (or original abstract if PDF extraction fails)
        - link_pdf: Direct link to PDF file
        - pdf_error: Error message if PDF fetch failed (optional)
        - text_error: Error message if text extraction failed (optional)

    Internal Behavior:
        _INCLUDE_PDF = True      # Download PDFs
        _EXTRACT_TEXT = True     # Extract text from PDFs
        _MAX_PAGES = 6           # Only extract first 6 pages (speed)
        _TEXT_CHARS = 5000       # Return first 5000 characters
        _SAVE_FULL_TEXT = False  # Don't save complete text (too long)
        _SLEEP_SECONDS = 1.0     # Delay between PDF downloads (rate limiting)

    Example:
        results = arxiv_search_tool("attention mechanism transformers", max_results=3)
        for paper in results:
            print(f"{paper['title']}: {paper['summary'][:100]}...")
    """
    # ========================================================================
    # CONFIGURATION FLAGS
    # ========================================================================
    _INCLUDE_PDF = True          # Download PDFs?
    _EXTRACT_TEXT = True         # Extract text from PDFs?
    _MAX_PAGES = 6              # Only read first 6 pages (for speed)
    _TEXT_CHARS = 5000          # Return first 5000 characters
    _SAVE_FULL_TEXT = False     # If True, return ALL text (slow, huge)
    _SLEEP_SECONDS = 1.0        # Delay between downloads (be polite)

    # ========================================================================
    # CALL ARXIV API
    # ========================================================================

    # Construct API query URL
    api_url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{requests.utils.quote(query)}"
        f"&start=0&max_results={max_results}"
    )

    out: List[Dict] = []

    try:
        resp = session.get(api_url, timeout=60)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return [{"error": f"arXiv API request failed: {e}"}]

    # ========================================================================
    # PARSE XML RESPONSE
    # ========================================================================

    try:
        root = ET.fromstring(resp.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}  # XML namespace

        # Process each paper in the results
        for entry in root.findall("atom:entry", ns):
            # Extract metadata
            title = (
                entry.findtext("atom:title", default="", namespaces=ns) or ""
            ).strip()

            published = (
                entry.findtext("atom:published", default="", namespaces=ns) or ""
            )[:10]  # Just YYYY-MM-DD

            url_abs = entry.findtext("atom:id", default="", namespaces=ns) or ""

            # Original abstract from arXiv
            abstract_summary = (
                entry.findtext("atom:summary", default="", namespaces=ns) or ""
            ).strip()

            # Extract authors
            authors = []
            for a in entry.findall("atom:author", ns):
                nm = a.findtext("atom:name", default="", namespaces=ns)
                if nm:
                    authors.append(nm)

            # Find PDF link (prefer explicit PDF link)
            link_pdf = None
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    link_pdf = link.attrib.get("href")
                    break

            # If no explicit PDF link, construct from abstract URL
            if not link_pdf and url_abs:
                link_pdf = ensure_pdf_url(url_abs)

            # Build result item with metadata
            item = {
                "title": title,
                "authors": authors,
                "published": published,
                "url": url_abs,
                "summary": abstract_summary,  # Will be overwritten if PDF extracted
                "link_pdf": link_pdf,
            }

            # ================================================================
            # DOWNLOAD AND EXTRACT PDF TEXT
            # ================================================================

            pdf_bytes = None
            if (_INCLUDE_PDF or _EXTRACT_TEXT) and link_pdf:
                try:
                    pdf_bytes = fetch_pdf_bytes(link_pdf, timeout=90)
                    time.sleep(_SLEEP_SECONDS)  # Rate limiting
                except Exception as e:
                    item["pdf_error"] = f"PDF fetch failed: {e}"

            # Extract text from PDF if we have it
            if _EXTRACT_TEXT and pdf_bytes:
                try:
                    text = pdf_bytes_to_text(pdf_bytes, max_pages=_MAX_PAGES)
                    text = clean_text(text) if text else ""

                    if text:
                        # Overwrite 'summary' with actual paper text
                        if _SAVE_FULL_TEXT:
                            item["summary"] = text  # Full text (huge!)
                        else:
                            item["summary"] = text[:_TEXT_CHARS]  # First 5000 chars
                except Exception as e:
                    item["text_error"] = f"Text extraction failed: {e}"

            out.append(item)

        return out

    except ET.ParseError as e:
        return [{"error": f"arXiv API XML parse failed: {e}"}]
    except Exception as e:
        return [{"error": f"Unexpected error: {e}"}]


# Tool definition for LLM function calling
# This tells the LLM what the function does and what parameters it accepts
arxiv_tool_def = {
    "type": "function",
    "function": {
        "name": "arxiv_search_tool",
        "description": "Searches arXiv and (internally) fetches PDFs to memory and extracts text.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for academic papers."
                },
                "max_results": {
                    "type": "integer",
                    "default": 3,
                    "description": "Maximum number of papers to return."
                },
            },
            "required": ["query"],
        },
    },
}


# ============================================================================
# TAVILY SEARCH TOOL - General Web Search
# ============================================================================

def tavily_search_tool(
    query: str,
    max_results: int = 5,
    include_images: bool = False
) -> list[dict]:
    """
    Performs a general-purpose web search using the Tavily API.

    Tavily is a search API optimized for LLM use cases. It returns
    clean, structured results from across the web including news sites,
    blogs, documentation, and other sources.

    Args:
        query: Search keywords (e.g., "latest AI developments 2024")
        max_results: Number of results to return (default 5)
        include_images: Whether to include image URLs in results

    Returns:
        List of dictionaries, each containing:
        - title: Page title
        - content: Relevant snippet/summary from the page
        - url: Link to the source
        - image_url: (if include_images=True) URL to related image

        On error, returns: [{"error": "error message"}]

    Environment Variables:
        - TAVILY_API_KEY: Required API key for Tavily service
        - DLAI_TAVILY_BASE_URL: Optional custom API endpoint

    Example:
        results = tavily_search_tool("GPT-4 capabilities", max_results=5)
        for result in results:
            print(f"{result['title']}: {result['url']}")
    """
    # Get API key from environment
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables.")

    # Create Tavily client (supports custom base URL for proxies)
    client = TavilyClient(
        api_key,
        api_base_url=os.getenv("DLAI_TAVILY_BASE_URL")
    )

    try:
        # Execute search
        response = client.search(
            query=query,
            max_results=max_results,
            include_images=include_images
        )

        # Format results
        results = []
        for r in response.get("results", []):
            results.append(
                {
                    "title": r.get("title", ""),
                    "content": r.get("content", ""),
                    "url": r.get("url", ""),
                }
            )

        # Add images if requested
        if include_images:
            for img_url in response.get("images", []):
                results.append({"image_url": img_url})

        return results

    except Exception as e:
        # Return error in LLM-friendly format
        return [{"error": str(e)}]


# Tool definition for LLM function calling
tavily_tool_def = {
    "type": "function",
    "function": {
        "name": "tavily_search_tool",
        "description": "Performs a general-purpose web search using the Tavily API.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for retrieving information from the web.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
                "include_images": {
                    "type": "boolean",
                    "description": "Whether to include image results.",
                    "default": False,
                },
            },
            "required": ["query"],
        },
    },
}


# ============================================================================
# WIKIPEDIA SEARCH TOOL - Encyclopedia Lookups
# ============================================================================

def wikipedia_search_tool(query: str, sentences: int = 5) -> List[Dict]:
    """
    Searches Wikipedia for a summary of the given query.

    Useful for:
    - Background information and context
    - Definitions of terms and concepts
    - Historical information
    - General overviews of topics

    Args:
        query: Search keywords (e.g., "machine learning")
        sentences: Number of sentences to include in summary (default 5)

    Returns:
        List with a single dictionary containing:
        - title: Wikipedia article title
        - summary: Summary text (N sentences from article intro)
        - url: Link to the Wikipedia article

        On error, returns: [{"error": "error message"}]

    How it works:
        1. Searches Wikipedia for matching articles
        2. Takes the top result
        3. Extracts the first N sentences as a summary
        4. Returns article details with link

    Example:
        results = wikipedia_search_tool("transformer neural network", sentences=3)
        print(results[0]['summary'])
    """
    try:
        # Search for matching articles, take first result
        page_title = wikipedia.search(query)[0]

        # Get the full article page
        page = wikipedia.page(page_title)

        # Extract summary (first N sentences)
        summary = wikipedia.summary(page_title, sentences=sentences)

        return [{
            "title": page.title,
            "summary": summary,
            "url": page.url
        }]

    except Exception as e:
        # Handle errors (disambiguation, no results, etc.)
        return [{"error": str(e)}]


# Tool definition for LLM function calling
wikipedia_tool_def = {
    "type": "function",
    "function": {
        "name": "wikipedia_search_tool",
        "description": "Searches for a Wikipedia article summary by query string.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords for the Wikipedia article.",
                },
                "sentences": {
                    "type": "integer",
                    "description": "Number of sentences in the summary.",
                    "default": 5,
                },
            },
            "required": ["query"],
        },
    },
}


# ============================================================================
# TOOL MAPPING - For Dynamic Tool Resolution
# ============================================================================

# This mapping allows dynamic lookup of tool functions by name
# Useful if you want to call tools based on string names
tool_mapping = {
    "tavily_search_tool": tavily_search_tool,
    "arxiv_search_tool": arxiv_search_tool,
    "wikipedia_search_tool": wikipedia_search_tool,
}
