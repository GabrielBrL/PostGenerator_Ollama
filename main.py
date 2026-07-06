from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import httpx
import json
import re
 
app = FastAPI(
    title="Post Content Pipeline",
    description=(
        "Three-agent pipeline: Topic Generator → Idea Developer → Post Writer.\n\n"
        "**Flow:** `/generate` produces topics → user picks one → `/pipeline` develops the idea and writes the final post."
    ),
    version="2.0.0",
)
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
 
OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "mistral:7b"
 
TOPIC_GENERATOR_PROMPT = """You are an expert in technical content marketing for software developers.
Your sole purpose is to generate creative, relevant, and strategic post topic ideas based on the technology stacks provided by the user.
 
When the user provides all required inputs, respond ONLY with a valid JSON array. No markdown fences, no extra text before or after the array.
 
Rules:
- Be creative: avoid generic titles. Favor unexpected or provocative angles that address real developer pain points.
- Vary the formats: step-by-step tutorial, thread, opinion piece, comparison, real-world case study, quick tip, post series, etc.
- Adapt tone and length to the chosen format and platform.
- Write all titles and hooks in the language specified by the user. If no language is specified, default to English.
 
Each object in the JSON array must contain:
- "title" (string, max 80 chars): attention-grabbing post title
- "hook" (string): opening hook / first paragraph with 2-3 compelling sentences in the requested language
- "platform" (string): one of LinkedIn | Technical Blog | Twitter/X | YouTube | Newsletter
- "stacks" (array of strings): relevant stacks for this specific topic
- "format" (string): suggested format e.g. "Step-by-step tutorial", "Thread", "Opinion piece", "Comparison", "Real-world case study", "Quick tip"
- "level" (string): one of beginner | intermediate | advanced
- "language" (string): language in which title and hook are written"""

IDEA_DEVELOPER_PROMPT = """You are an expert technical content strategist specialized in helping software developers create highly engaging and educational content.
 
Your sole purpose is to transform a provided topic into a complete content idea ready to be written or recorded.
 
The user will provide a JSON object containing: title, hook, platform, stacks, format, level, language.
 
Your job is to expand this topic into a more actionable and structured content idea.
 
Respond ONLY with a valid JSON object. No markdown fences and no additional explanations.
 
The JSON object must contain:
- "title" (string): Improved or refined version of the original title, if necessary
- "content_angle" (string): The main perspective or unique angle of the content. Explain WHY this topic is interesting and what pain point it solves.
- "target_audience" (string): Who this content is for.
- "content_goal" (string): One of: educate | entertain | inspire | provoke discussion | generate leads | build authority
- "outline" (array of strings): A step-by-step structure for the content.
- "key_points" (array of strings): Important technical insights or lessons the content should communicate.
- "cta" (string): Suggested call-to-action aligned with the platform and format.
- "suggested_visuals" (array of strings): Ideas for visuals, diagrams, code screenshots, memes, architecture drawings, charts, etc.
- "estimated_length" (string): Suggested size depending on platform. Examples: "5-slide carousel", "8-minute video", "1200-word article", "10-tweet thread"
- "tone" (string): Recommended tone. Examples: professional | opinionated | educational | storytelling | provocative | casual technical
- "language" (string): Same language provided by the user
 
Rules:
- Adapt the structure to the platform and format.
- Focus on real-world developer problems and practical insights.
- Avoid generic outlines.
- Keep all generated text in the requested language.
- Do not generate the full post itself. Only generate the strategic content idea and structure."""
 
POST_WRITER_PROMPT = """You are an expert technical content writer for software developers.
 
You receive a JSON object with the complete strategic brief for a post. Write the full, publish-ready post based on that brief.
 
The input JSON contains: title, content_angle, target_audience, content_goal, outline, key_points, cta, suggested_visuals, estimated_length, tone, language, platform, format, stacks, level.
 
Writing rules:
- Follow the outline order exactly.
- Cover every key_point.
- Use the provided cta verbatim or adapt it minimally.
- Match the tone throughout.
- Write at the depth appropriate for the given level.
- Write entirely in the specified language.
- Output only the finished post — no preamble, no meta-commentary, no JSON wrapper.
 
Platform formatting rules:
 
LINKEDIN:
- 100% copy-paste ready. No placeholders, no markdown, no HTML.
- Max 3000 characters total.
- First line = punchy hook. No greeting.
- One blank line between every paragraph. 1-3 lines per paragraph.
- Emphasis: ALL CAPS for one word max once or twice.
- Bullets: use →, •, or ✅
- Do NOT include suggested_visuals in the text.
- Close with cta then hashtags (3-5 tags, last line).
- No carousel-style content.
 
TECHNICAL BLOG:
- Full Markdown: # title, ## sections, triple backticks for code.
- Complete runnable code with comments.
- Visuals as `> 📊 Visual: <description>` blockquotes.
- Close with ## Summary then cta.
 
TWITTER/X:
- Numbered thread: 1/, 2/, 3/ ...
- Max 280 chars per tweet. Max 12 tweets.
- Tweet 1 = hook. Last tweet = cta.
- Visuals as [image: <description>] inline.
 
YOUTUBE:
- Full spoken script.
- ## 0:00 HOOK, ## 0:30 INTRO, ## MM:SS <Section> per outline item, ## OUTRO.
- Use [PAUSE] and [B-ROLL: <visual>] markers.
 
NEWSLETTER:
- Top block: Subject / Preview / TL;DR
- Body: personal opener → outline sections → takeaway → closing.
- Warm conversational tone. Personal sign-off."""
 
 
# ── Models ──────────────────────────────────────────────────────────────────
 
class TopicItem(BaseModel):
    title: str
    hook: str
    platform: str
    stacks: List[str]
    format: str
    level: str
    language: str

class ContentIdea(BaseModel):
    title: str
    content_angle: str
    target_audience: str
    content_goal: str
    outline: List[str]
    key_points: List[str]
    cta: str
    suggested_visuals: List[str]
    estimated_length: str
    tone: str
    language: str

class GenerateRequest(BaseModel):
    stacks: List[str] = Field(..., example=["Node.js", "React", "PostgreSQL"])
    platform: str = Field(..., example="LinkedIn")
    level: str = Field(..., example="intermediate")
    quantity: int = Field(5, ge=1, le=5, description="Number of topics (1-5)")
    language: str = Field("English", example="English")
    model: Optional[str] = Field(DEFAULT_MODEL, description="Ollama model name")

class PipelineRequest(BaseModel):
    topic: TopicItem = Field(..., description="A topic object returned by /generate")
    model: Optional[str] = Field(None, description="Ollama model name")
 
class GenerateResponse(BaseModel):
    topics: List[TopicItem]
 
class OllamaStatus(BaseModel):
    connected: bool
    models: List[str]
    message: str
 
 
# ── Helpers ──────────────────────────────────────────────────────────────────
 
async def call_ollama(system: str, messages: list, model: str) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=300) as client:
        try:
            resp = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except httpx.ConnectError:
            raise HTTPException(503, "Cannot reach Ollama. Run: ollama serve")
        except httpx.HTTPStatusError as e:
            raise HTTPException(502, f"Ollama error {e.response.status_code}: {e.response.text}")
 
 
STACK_ALIASES = ["stacks", "technologies", "tech_stack", "stack", "tools", "techStack"]
PLATFORM_ALIASES = ["platform", "platforms", "target_platform", "targetPlatform"]
FORMAT_ALIASES = ["format", "content_format", "type", "post_type", "contentFormat"]
LEVEL_ALIASES = ["level", "difficulty", "content_level", "contentLevel"]
LANGUAGE_ALIASES = ["language", "lang", "content_language", "contentLanguage"]
 
 
def _first(d: dict, keys: list, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default
 
def _coerce_str(v, default="") -> str:
    if isinstance(v, list): v = v[0] if v else default
    return str(v).strip() if v is not None else default
 
def _coerce_list(v) -> List[str]:
    if isinstance(v, list): return [str(i).strip() for i in v]
    if isinstance(v, str):  return [s.strip() for s in re.split(r"[,;]", v) if s.strip()]
    return []
 
def _extract_json_object(text: str) -> Optional[dict]:
    clean = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\{[\s\S]*\}", clean)
    if match:
        return json.loads(match.group())
    return None
 
def _extract_json_array(text: str) -> Optional[list]:
    clean = re.sub(r"```json|```", "", text).strip()
    match = re.search(r"\[[\s\S]*\]", clean)
    if match:
        data = json.loads(match.group())
        if isinstance(data, list) and data and "title" in data[0]:
            return data
    return None
 
def normalize_topic(raw: dict) -> Optional[dict]:
    try:
        return {
            "title":    _coerce_str(raw.get("title")),
            "hook":     _coerce_str(raw.get("hook")),
            "platform": _coerce_str(_first(raw, PLATFORM_ALIASES)),
            "stacks":   _coerce_list(_first(raw, STACK_ALIASES, [])),
            "format":   _coerce_str(_first(raw, FORMAT_ALIASES)),
            "level":    _coerce_str(_first(raw, LEVEL_ALIASES), "intermediate"),
            "language": _coerce_str(_first(raw, LANGUAGE_ALIASES), "English"),
        }
    except Exception:
        return None
 
def normalize_idea(raw: dict, topic: TopicItem) -> dict:
    return {
        "title":             _coerce_str(raw.get("title") or topic.title),
        "content_angle":     _coerce_str(raw.get("content_angle")),
        "target_audience":   _coerce_str(raw.get("target_audience")),
        "content_goal":      _coerce_str(raw.get("content_goal")),
        "outline":           _coerce_list(raw.get("outline", [])),
        "key_points":        _coerce_list(raw.get("key_points", [])),
        "cta":               _coerce_str(raw.get("cta")),
        "suggested_visuals": _coerce_list(raw.get("suggested_visuals", [])),
        "estimated_length":  _coerce_str(raw.get("estimated_length")),
        "tone":              _coerce_str(raw.get("tone")),
        "language":          _coerce_str(raw.get("language") or topic.language),
    }


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=OllamaStatus, tags=["Status"])
async def health():
    """Check Ollama connectivity and list available models."""
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            resp = await client.get(f"{OLLAMA_URL}/api/tags")
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return OllamaStatus(connected=True, models=models, message="Ollama is running.")
        except Exception as e:
            return OllamaStatus(connected=False, models=[], message=str(e))


@app.post("/generate-topic", response_model=GenerateResponse, tags=["Agent"])
async def generate(req: GenerateRequest):
    """
    **Agent 1 — Topic Generator**
 
    Provide stacks, platform, level, count and language.
    Returns a list of topic ideas — pick one and send it to `/pipeline`.
    """
    model = req.model or DEFAULT_MODEL
    user_message = (
        f"Generate {req.quantity} post topic ideas with the following details:\n"
        f"- Technology stacks: {', '.join(req.stacks)}\n"
        f"- Target platform: {req.platform}\n"
        f"- Content level: {req.level}\n"
        f"- Language: {req.language}\n\n"
        "Respond ONLY with the JSON array, no extra text."
    )
    reply = await call_ollama(TOPIC_GENERATOR_PROMPT, [{"role": "user", "content": user_message}], model)
    raw_list = _extract_json_array(reply)
    if not raw_list:
        raise HTTPException(422, f"Model did not return a valid JSON array. Raw: {reply[:400]}")
    topicsNormalized = [normalize_topic(t) for t in raw_list]
    return GenerateResponse(topics=[TopicItem(**t) for t in topicsNormalized if t])

@app.post("/generate-post", response_model=str, tags=["2 · Full Pipeline"])
async def pipeline(req: PipelineRequest):
    """
    **Agents 2 + 3 — Idea Developer → Post Writer**
 
    Send a topic object from `/generate`. The pipeline:
    1. Expands it into a full content brief (Idea Developer)
    2. Writes the complete publish-ready post (Post Writer)
 
    Returns `topic`, `idea`, and `post`.
    """
    model = req.model or DEFAULT_MODEL
    topic = req.topic
 
    # ── Agent 2: Idea Developer ───────────────────────────────────────────────
    idea_reply = await call_ollama(
        IDEA_DEVELOPER_PROMPT,
        [{"role": "user", "content": json.dumps(topic.model_dump(), ensure_ascii=False)}],
        model,
    )
    try:
        idea_raw = _extract_json_object(idea_reply)
        if not idea_raw:
            raise ValueError("no JSON object")
    except Exception:
        raise HTTPException(422, f"Idea Developer did not return valid JSON. Raw: {idea_reply[:400]}")
 
    idea_dict = normalize_idea(idea_raw, topic)
    idea = ContentIdea(**idea_dict)
 
    # ── Agent 3: Post Writer ──────────────────────────────────────────────────
    writer_input = json.dumps({
        **idea_dict,
        "platform": topic.platform,
        "format":   topic.format,
        "stacks":   topic.stacks,
        "level":    topic.level,
    }, ensure_ascii=False)
 
    post_reply = await call_ollama(
        POST_WRITER_PROMPT,
        [{"role": "user", "content": writer_input}],
        model,
    )
 
    return post_reply.strip().replace("\\n", "\n")