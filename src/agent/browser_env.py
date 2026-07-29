"""
Browser Environment - Provides observation extraction across three modes.

Extracts the Accessibility Tree, DOM, and screenshots from web pages
using Playwright, formatted for LLM consumption.
"""

import asyncio
import base64
from dataclasses import dataclass

from playwright.async_api import Page, async_playwright


@dataclass
class Observation:
    """A single observation of a web page in a specific mode."""
    url: str
    mode: str  # "accessibility_tree", "dom", "screenshot"
    content: str  # Text content (AXTree/DOM) or base64 screenshot
    token_estimate: int  # Rough token count (~4 chars per token)


async def extract_accessibility_tree(page: Page) -> str:
    """
    Extract the Accessibility Tree from a Playwright page via CDP.
    Returns a text representation suitable for LLM consumption.
    """
    client = await page.context.new_cdp_session(page)
    tree = await client.send("Accessibility.getFullAXTree")
    await client.detach()
    nodes = tree.get("nodes", [])
    if not nodes:
        return "[Empty accessibility tree]"
    return _format_cdp_ax_tree(nodes)


def _format_cdp_ax_tree(nodes: list[dict]) -> str:
    """Format CDP Accessibility.getFullAXTree nodes into readable text."""
    # Build parent-child mapping
    children_map: dict[str, list[str]] = {}
    node_map: dict[str, dict] = {}
    for node in nodes:
        nid = node.get("nodeId", "")
        node_map[nid] = node
        for child_id in node.get("childIds", []):
            children_map.setdefault(nid, []).append(child_id)

    lines: list[str] = []
    root_id = nodes[0]["nodeId"] if nodes else None

    def walk(nid: str, depth: int = 0) -> None:
        node = node_map.get(nid)
        if not node:
            return
        role = node.get("role", {}).get("value", "none")
        name = node.get("name", {}).get("value", "")
        # Skip invisible/ignored nodes
        if node.get("ignored", False):
            for cid in children_map.get(nid, []):
                walk(cid, depth)
            return
        # InlineTextBox duplicates its StaticText parent's content — always skip
        if role == "InlineTextBox" or (role in ("none", "generic") and not name):
            for cid in children_map.get(nid, []):
                walk(cid, depth)
            return

        prefix = "  " * depth
        parts = [f"{prefix}[{role}]"]
        if name:
            parts.append(f'"{name[:120]}"')
        # Add relevant properties
        for prop in node.get("properties", []):
            pname = prop.get("name", "")
            pval = prop.get("value", {}).get("value", "")
            if pname in ("checked", "expanded", "selected", "disabled", "pressed") and pval:
                parts.append(f"{pname}={pval}")
        lines.append(" ".join(parts))
        for cid in children_map.get(nid, []):
            walk(cid, depth + 1)

    if root_id:
        walk(root_id)
    text = "\n".join(lines)
    max_length = 60000  # ~15k tokens; keeps pathological pages bounded
    if len(text) > max_length:
        text = text[:max_length] + f"\n... [truncated at {max_length} chars]"
    return text


async def extract_dom(page: Page, max_length: int = 50000) -> str:
    """
    Extract a simplified DOM representation from a Playwright page.
    Strips scripts, styles, and excessive whitespace.

    Retries once on "Execution context was destroyed", which occurs when the
    page navigates mid-evaluation; waiting for the DOM to settle resolves it.
    """
    for attempt in range(2):
        try:
            return await _extract_dom_once(page, max_length)
        except Exception as e:
            if attempt == 0 and "Execution context was destroyed" in str(e):
                try:
                    await page.wait_for_load_state("domcontentloaded", timeout=8000)
                except Exception:
                    await page.wait_for_timeout(1500)
                continue
            raise
    return "[DOM extraction failed]"


async def _extract_dom_once(page: Page, max_length: int = 50000) -> str:
    dom = await page.evaluate("""
        () => {
            // Clone document and strip non-content elements
            const clone = document.documentElement.cloneNode(true);
            const remove = clone.querySelectorAll('script, style, noscript, svg, link[rel=stylesheet]');
            remove.forEach(el => el.remove());

            // Get outer HTML
            let html = clone.outerHTML;

            // Collapse whitespace
            html = html.replace(/\\s+/g, ' ');
            return html;
        }
    """)
    if len(dom) > max_length:
        dom = dom[:max_length] + f"\n... [truncated at {max_length} chars]"
    return dom


async def extract_screenshot(page: Page) -> str:
    """
    Take a viewport screenshot and return as base64 string.

    Animations are disabled and a bounded timeout is used because busy
    pages (e.g. carousels, streaming ads) never reach render stability
    and would otherwise time out at Playwright's default.
    """
    try:
        screenshot_bytes = await page.screenshot(
            full_page=False, type="png", animations="disabled", timeout=20000
        )
    except Exception:
        # Last resort: capture whatever is painted right now
        screenshot_bytes = await page.screenshot(
            full_page=False, type="png", animations="allow", timeout=10000
        )
    return base64.b64encode(screenshot_bytes).decode("utf-8")


async def get_observation(page: Page, mode: str) -> Observation:
    """
    Get an observation from the current page in the specified mode.

    Args:
        page: Playwright page object.
        mode: One of "accessibility_tree", "dom", "screenshot".

    Returns:
        Observation with content and metadata.
    """
    url = page.url

    if mode == "accessibility_tree":
        content = await extract_accessibility_tree(page)
    elif mode == "dom":
        content = await extract_dom(page)
    elif mode == "screenshot":
        content = await extract_screenshot(page)
    else:
        raise ValueError(f"Unknown observation mode: {mode}")

    token_estimate = len(content) // 4  # rough estimate

    return Observation(
        url=url,
        mode=mode,
        content=content,
        token_estimate=token_estimate,
    )
