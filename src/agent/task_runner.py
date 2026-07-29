"""
Agent Task Execution Module - LLM-based web agent task runner.

Executes defined tasks on websites using an LLM agent that observes
the page via Accessibility Tree, DOM, or screenshot, then takes actions.

Design notes (documented in thesis Ch.5):
- Bounded observation context: each LLM call receives the task, a compact
  action history (action -> outcome per step), and ONLY the current
  observation. Prior observations are not resent, keeping cost and context
  bounded regardless of step count (cf. AgentOccam-style observation
  refinement).
- Dual success measurement: the agent self-reports via done(success=...),
  and the runner independently verifies completion by keyword matching
  against the final URL and visible page text (verified_success).
- Token accounting: real input/output token counts are taken from the API
  usage field, not estimated.
"""

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum

from playwright.async_api import Page, async_playwright

from .browser_env import Observation, get_observation


class TaskStatus(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass
class TaskStep:
    """A single step in task execution."""
    step_number: int
    observation_tokens: int
    action: str
    reasoning: str
    outcome: str
    timestamp: float


@dataclass
class TaskResult:
    """Result of executing a single task on a website."""
    url: str
    task_description: str
    observation_mode: str
    status: TaskStatus
    task_id: str = ""
    category: str = ""
    run_index: int = 0
    llm_model: str = ""
    steps: list[TaskStep] = field(default_factory=list)
    total_steps: int = 0
    error_message: str = ""
    final_url: str = ""
    verified_success: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def success(self) -> bool:
        return self.status == TaskStatus.SUCCESS

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "task": self.task_description,
            "task_id": self.task_id,
            "category": self.category,
            "run_index": self.run_index,
            "observation_mode": self.observation_mode,
            "llm_model": self.llm_model,
            "status": self.status.value,
            "verified_success": self.verified_success,
            "final_url": self.final_url,
            "total_steps": self.total_steps,
            "duration_seconds": round(self.duration, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error_message,
            "steps": [
                {
                    "step": s.step_number,
                    "action": s.action,
                    "reasoning": s.reasoning,
                    "outcome": s.outcome,
                    "observation_tokens": s.observation_tokens,
                }
                for s in self.steps
            ],
        }


@dataclass
class WebTask:
    """Definition of a task to execute on a website."""
    url: str
    description: str
    success_criteria: str  # How to verify task completion
    max_steps: int = 20
    task_id: str = ""
    category: str = ""
    success_keywords: list[str] = field(default_factory=list)


# Available browser actions the agent can take
AVAILABLE_ACTIONS = """Available actions:
- click(selector): Click an element. Use text content or role, e.g. click("Sign In") or click("[role=button][name=Submit]")
- type(selector, text): Type text into an input field, e.g. type("Search", "machine learning")
- press_enter(): Press the Enter key (e.g. to submit a search after typing)
- scroll(direction): Scroll the page. direction is "up" or "down"
- navigate(url): Navigate to a URL
- wait(seconds): Wait for a number of seconds
- done(success, reason): End the task. success is true/false, reason explains why.
"""

SYSTEM_PROMPT = """You are a web navigation agent. You observe web pages and take actions to complete tasks.

{actions}

You will receive the task, a history of your previous actions with their outcomes, and an observation of the current page state.
Respond with a JSON object containing:
- "reasoning": Brief explanation of what you see and your plan
- "action": The action to take (one of the available actions above)

Important:
- Only take ONE action per step
- If a previous action failed, try a different selector or approach
- Dismiss cookie/consent banners if they block the content you need
- After typing into a search box, use press_enter() to submit the search
- Use the done() action when you've completed the task or determined it cannot be completed
- Be precise with selectors — match text content or accessibility roles exactly
- Inside the action string, use single quotes for arguments, e.g. click('Sign In') or type('Search', 'gift') — never nest double quotes"""


# Scripted responses for the mock provider (dry runs / plumbing tests).
_MOCK_SCRIPT = [
    '{"reasoning": "Dry run: scrolling to inspect the page.", "action": "scroll(down)"}',
    '{"reasoning": "Dry run complete.", "action": "done(false, \'mock provider — no LLM\')"}',
]


DEFAULT_MODEL = "claude-sonnet-5"


async def _call_llm(
    messages: list[dict],
    provider: str = "anthropic",
    model: str = DEFAULT_MODEL,
    max_retries: int = 3,
) -> tuple[str, dict]:
    """
    Call an LLM API to get the next action.

    Returns:
        (response_text, usage) where usage has input_tokens/output_tokens.
    """
    if provider == "mock":
        # Deterministic scripted agent for dry runs — no API key needed.
        step_index = sum(1 for m in messages if m["role"] == "assistant")
        text = _MOCK_SCRIPT[min(step_index, len(_MOCK_SCRIPT) - 1)]
        return text, {"input_tokens": 0, "output_tokens": 0}

    if provider != "anthropic":
        raise ValueError(f"Unsupported LLM provider: {provider}")

    import os

    import httpx

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Export it or add to framework/.env"
        )

    # Claude Sonnet 5 (and Opus 4.7+) reject sampling parameters like
    # temperature, and run adaptive thinking by default; the agent takes one
    # small action per call, so thinking is disabled to keep each step fast
    # and cheap. Sampling defaults apply (documented in thesis Ch.5).
    payload: dict = {
        "model": model,
        "max_tokens": 1024,
        "system": SYSTEM_PROMPT.format(actions=AVAILABLE_ACTIONS),
        "messages": messages,
    }
    if "sonnet-5" in model or "opus-4-7" in model or "opus-4-8" in model:
        payload["thinking"] = {"type": "disabled"}

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json=payload,
                    timeout=120.0,
                )
                if resp.status_code in (429, 500, 502, 503, 529):
                    raise httpx.HTTPStatusError(
                        f"Retryable status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                return data["content"][0]["text"], {
                    "input_tokens": usage.get("input_tokens", 0),
                    "output_tokens": usage.get("output_tokens", 0),
                }
        except (httpx.TimeoutException, httpx.TransportError, httpx.HTTPStatusError) as e:
            if isinstance(e, httpx.HTTPStatusError) and e.response is not None:
                if e.response.status_code not in (429, 500, 502, 503, 529):
                    # Non-retryable: surface the API's error body, not just
                    # the status line — essential for diagnosing 400s
                    raise RuntimeError(
                        f"API {e.response.status_code}: {e.response.text[:300]}"
                    ) from e
            last_error = e
            if attempt < max_retries:
                await asyncio.sleep(2 * (attempt + 1) ** 2)  # 2s, 8s, 18s
    raise RuntimeError(f"LLM call failed after {max_retries + 1} attempts: {last_error}")


# Matches a single action call on one line, e.g. click('Sign In')
_ACTION_CALL_RE = re.compile(
    r"(?:click|type|press_enter|scroll|navigate|wait|done)\([^\n]*\)"
)


def _parse_action(response_text: str) -> tuple[str, str]:
    """
    Parse the LLM response to extract action and reasoning.

    Falls back to regex extraction when the JSON is malformed — models
    sometimes nest unescaped double quotes inside the action string or emit
    the action value unquoted, which breaks strict JSON parsing.
    """
    text = response_text.strip()
    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
        action = data.get("action", "done(false, 'No action parsed')")
        if isinstance(action, dict):
            # Structured action object, e.g. {"name": "click", "selector": "X"}
            name = action.get("name") or action.get("type") or "done"
            args = [str(v) for k, v in action.items() if k not in ("name", "type")]
            action = f"{name}({', '.join(repr(a) for a in args)})"
        return str(action), str(data.get("reasoning", ""))
    except (json.JSONDecodeError, IndexError):
        pass

    # Malformed JSON: extract the action call after the "action" key if
    # present (reasoning prose may also mention action-like text), else the
    # last action-shaped call anywhere in the response.
    idx = text.find('"action"')
    match = _ACTION_CALL_RE.search(text[idx:] if idx != -1 else text)
    if match is None:
        matches = _ACTION_CALL_RE.findall(text)
        match_text = matches[-1] if matches else None
    else:
        match_text = match.group(0)

    reasoning_m = re.search(r'"reasoning"\s*:\s*"((?:[^"\\]|\\.)*)', text)
    reasoning = reasoning_m.group(1)[:300] if reasoning_m else text[:200]

    if match_text:
        return match_text, reasoning
    return "done(false, 'Failed to parse LLM response')", response_text[:200]


async def _execute_action(page: Page, action: str) -> tuple[bool, str]:
    """
    Execute a browser action on the page.

    Returns:
        (is_done, message) — is_done=True if the task should end.
    """
    action = action.strip()

    if action.startswith("done("):
        return True, action

    if action.startswith("click("):
        target = action[6:-1].strip().strip('"').strip("'")
        try:
            await page.get_by_role("link", name=target).first.click(timeout=5000)
        except Exception:
            try:
                await page.get_by_role("button", name=target).first.click(timeout=5000)
            except Exception:
                try:
                    await page.get_by_text(target, exact=False).first.click(timeout=5000)
                except Exception:
                    try:
                        await page.locator(target).first.click(timeout=5000)
                    except Exception as e:
                        return False, f"Click failed: {str(e).splitlines()[0][:160]}"
        await page.wait_for_timeout(1500)
        return False, f"Clicked: {target}"

    if action.startswith("type("):
        # Parse type("selector", "text") — quote-aware so selectors that
        # contain commas are not split apart
        m = re.match(
            r"""type\(\s*["'](.+?)["']\s*,\s*["'](.*)["']\s*\)\s*$""",
            action,
            re.DOTALL,
        )
        if m:
            parts = [m.group(1), m.group(2)]
        else:
            parts = action[5:-1].split(",", 1)
        if len(parts) == 2:
            selector = parts[0].strip().strip('"').strip("'")
            text = parts[1].strip().strip('"').strip("'")
            try:
                await page.get_by_role("textbox", name=selector).first.fill(text, timeout=5000)
            except Exception:
                try:
                    await page.get_by_placeholder(selector).first.fill(text, timeout=5000)
                except Exception:
                    try:
                        await page.get_by_role("searchbox").first.fill(text, timeout=5000)
                    except Exception as e:
                        return False, f"Type failed: {str(e).splitlines()[0][:160]}"
            return False, f"Typed '{text}' into {selector}"
        return False, "Type action parse error"

    if action.startswith("press_enter"):
        await page.keyboard.press("Enter")
        await page.wait_for_timeout(1500)
        return False, "Pressed Enter"

    if action.startswith("scroll("):
        direction = action[7:-1].strip().strip('"').strip("'")
        delta = -500 if direction == "up" else 500
        await page.mouse.wheel(0, delta)
        await page.wait_for_timeout(500)
        return False, f"Scrolled {direction}"

    if action.startswith("navigate("):
        url = action[9:-1].strip().strip('"').strip("'")
        try:
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")
        except Exception as e:
            return False, f"Navigate failed: {str(e).splitlines()[0][:160]}"
        await page.wait_for_timeout(1500)
        return False, f"Navigated to {url}"

    if action.startswith("wait("):
        try:
            seconds = float(action[5:-1].strip())
        except ValueError:
            return False, "Wait action parse error"
        await page.wait_for_timeout(min(seconds * 1000, 5000))
        return False, f"Waited {seconds}s"

    return False, f"Unknown action: {action}"


def _build_step_messages(
    task: WebTask,
    obs: Observation,
    history: list[tuple[str, str]],
    step_num: int,
    observation_mode: str,
) -> list[dict]:
    """
    Build the message list for one LLM call: task + compact action history
    + current observation only. Prior observations are never resent.
    """
    history_text = (
        "\n".join(f"Step {i + 1}: {a} -> {o}" for i, (a, o) in enumerate(history))
        or "(no actions taken yet)"
    )
    header = (
        f"Task: {task.description}\n"
        f"Success criteria: {task.success_criteria}\n\n"
        f"Step {step_num}/{task.max_steps}. Current URL: {obs.url}\n\n"
        f"Previous actions and outcomes:\n{history_text}\n"
    )

    if observation_mode == "screenshot":
        content: str | list = [
            {"type": "text", "text": header + "\nHere is a screenshot of the current page:"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": obs.content}},
        ]
    else:
        content = header + f"\nCurrent page observation ({observation_mode}):\n{obs.content}"

    return [{"role": "user", "content": content}]


async def _verify_success(page: Page, task: WebTask, landed_url: str) -> tuple[bool, str]:
    """
    Independently verify task completion: the agent must have navigated
    away from the landing URL (post-redirect start page) AND a success
    keyword must appear in the final URL or visible page text. URL-change
    alone is not success; keyword match on the unchanged homepage is not
    success either (homepages often contain the keywords in nav link text).
    """
    final_url = page.url
    if not task.success_keywords:
        return False, final_url

    def norm(u: str) -> str:
        return u.rstrip("/").removeprefix("https://").removeprefix("http://").removeprefix("www.")

    if norm(final_url) in (norm(task.url), norm(landed_url)):
        return False, final_url
    try:
        page_text = await page.evaluate(
            "() => (document.body ? document.body.innerText : '').slice(0, 8000)"
        )
    except Exception:
        page_text = ""
    haystack = (final_url + "\n" + page_text).lower()
    verified = any(kw.lower() in haystack for kw in task.success_keywords)
    return verified, final_url


async def run_task(
    task: WebTask,
    observation_mode: str = "accessibility_tree",
    llm_provider: str = "anthropic",
    llm_model: str = DEFAULT_MODEL,
    run_index: int = 0,
) -> TaskResult:
    """
    Execute a web task using an LLM agent.

    Args:
        task: The task definition.
        observation_mode: How the agent observes the page.
        llm_provider: LLM API provider ("anthropic" or "mock").
        llm_model: Model identifier.
        run_index: Repetition index for repeated runs.

    Returns:
        TaskResult with execution details.
    """
    result = TaskResult(
        url=task.url,
        task_description=task.description,
        observation_mode=observation_mode,
        status=TaskStatus.ERROR,
        task_id=task.task_id,
        category=task.category,
        run_index=run_index,
        llm_model=llm_model if llm_provider != "mock" else "mock",
        start_time=time.time(),
    )

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 720},
        )
        page = await context.new_page()

        try:
            await page.goto(task.url, timeout=30000, wait_until="domcontentloaded")
            await page.wait_for_timeout(2000)
            landed_url = page.url  # post-redirect baseline for verification

            # Compact (action, outcome) history — mock provider counts
            # assistant turns, so keep a parallel message list for it.
            history: list[tuple[str, str]] = []
            mock_turns: list[dict] = []

            for step_num in range(1, task.max_steps + 1):
                obs = await get_observation(page, observation_mode)
                messages = _build_step_messages(
                    task, obs, history, step_num, observation_mode
                )
                response_text, usage = await _call_llm(
                    messages + mock_turns if llm_provider == "mock" else messages,
                    provider=llm_provider,
                    model=llm_model,
                )
                result.input_tokens += usage["input_tokens"]
                result.output_tokens += usage["output_tokens"]

                action, reasoning = _parse_action(response_text)

                # Execute action and feed the outcome back into history
                is_done, action_outcome = await _execute_action(page, action)
                history.append((action, action_outcome))
                if llm_provider == "mock":
                    mock_turns.append({"role": "assistant", "content": response_text})

                result.steps.append(TaskStep(
                    step_number=step_num,
                    observation_tokens=obs.token_estimate,
                    action=action,
                    reasoning=reasoning,
                    outcome=action_outcome,
                    timestamp=time.time(),
                ))

                if is_done:
                    if "true" in action.lower():
                        result.status = TaskStatus.SUCCESS
                    else:
                        result.status = TaskStatus.FAILURE
                        result.error_message = action_outcome
                    break
            else:
                result.status = TaskStatus.TIMEOUT
                result.error_message = f"Exceeded max steps ({task.max_steps})"

            result.verified_success, result.final_url = await _verify_success(page, task, landed_url)

        except Exception as e:
            result.status = TaskStatus.ERROR
            result.error_message = str(e)
            try:
                result.final_url = page.url
            except Exception:
                pass
        finally:
            result.total_steps = len(result.steps)
            result.end_time = time.time()
            await browser.close()

    return result


async def run_task_multi_mode(
    task: WebTask,
    modes: list[str] | None = None,
    runs_per_mode: int = 1,
    **kwargs,
) -> list[TaskResult]:
    """Run a task across multiple observation modes for comparison."""
    if modes is None:
        modes = ["accessibility_tree", "dom", "screenshot"]

    results = []
    for mode in modes:
        for run in range(runs_per_mode):
            result = await run_task(task, observation_mode=mode, run_index=run, **kwargs)
            results.append(result)
    return results
