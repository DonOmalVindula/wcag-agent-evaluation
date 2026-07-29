"""
Unit tests for the agent execution harness.

Covers the LLM response parser (including the malformed-response shapes
observed in the pilot study), action execution parsing, per-site task
generation, and the independent success-verification logic. All tests are
offline: no browser, no API calls.
"""

import pytest

from src.agent.task_runner import (
    WebTask,
    _build_step_messages,
    _execute_action,
    _parse_action,
    _verify_success,
)
from src.agent.task_templates import (
    SECTOR_PARAMS,
    SiteRecord,
    generate_all_tasks,
    tasks_for_site,
)
from src.agent.browser_env import Observation


# ---------------------------------------------------------------------------
# _parse_action — including the exact malformed shapes from the pilot
# ---------------------------------------------------------------------------

class TestParseAction:
    def test_well_formed_json(self):
        action, reasoning = _parse_action(
            '{"reasoning": "ok", "action": "click(\'Sign In\')"}'
        )
        assert action == "click('Sign In')"
        assert reasoning == "ok"

    def test_nested_double_quotes_in_action(self):
        # Pilot failure case: unescaped quotes inside the action string
        raw = '{"reasoning": "submit", "action": "click("[role=button][name=Go]")"}'
        action, _ = _parse_action(raw)
        assert action == 'click("[role=button][name=Go]")'

    def test_unquoted_action_value(self):
        # Pilot failure case: action emitted as a bare call, not a JSON string
        raw = '{"reasoning": "search", "action": type("#box", "gift")}'
        action, _ = _parse_action(raw)
        assert action == 'type("#box", "gift")'

    def test_structured_action_object(self):
        # Pilot crash case: action returned as a JSON object
        raw = '{"reasoning": "x", "action": {"name": "click", "selector": "Accept"}}'
        action, _ = _parse_action(raw)
        assert action == "click('Accept')"
        assert isinstance(action, str)

    def test_markdown_fenced_json(self):
        raw = '```json\n{"reasoning": "r", "action": "press_enter()"}\n```'
        action, _ = _parse_action(raw)
        assert action == "press_enter()"

    def test_done_with_reason(self):
        raw = '{"reasoning": "finished", "action": "done(true, \'found page\')"}'
        action, _ = _parse_action(raw)
        assert action.startswith("done(true")

    def test_unparseable_garbage_returns_done_false(self):
        action, _ = _parse_action("complete nonsense with no action at all")
        assert action.startswith("done(false")

    def test_action_always_string(self):
        for raw in ['{"action": 42}', '{"action": null}', '{"action": ["a"]}']:
            action, _ = _parse_action(raw)
            assert isinstance(action, str)


# ---------------------------------------------------------------------------
# _execute_action — argument parsing against a fake Playwright page
# ---------------------------------------------------------------------------

class _FakeLocator:
    def __init__(self, page, kind, target):
        self.page, self.kind, self.target = page, kind, target

    @property
    def first(self):
        return self

    async def fill(self, text, timeout=None):
        self.page.filled.append((self.target, text))

    async def click(self, timeout=None):
        self.page.clicked.append(self.target)


class _FakePage:
    """Minimal stand-in for playwright.async_api.Page."""

    def __init__(self, url="https://example.com", body_text=""):
        self.url = url
        self._body_text = body_text
        self.filled: list = []
        self.clicked: list = []
        self.pressed: list = []
        self.scrolled: list = []

    def get_by_role(self, role, name=None):
        return _FakeLocator(self, role, name)

    def get_by_placeholder(self, text):
        return _FakeLocator(self, "placeholder", text)

    def get_by_text(self, text, exact=False):
        return _FakeLocator(self, "text", text)

    def locator(self, sel):
        return _FakeLocator(self, "css", sel)

    async def wait_for_timeout(self, ms):
        pass

    async def evaluate(self, script):
        return self._body_text

    class _Kbd:
        def __init__(self, page):
            self.page = page

        async def press(self, key):
            self.page.pressed.append(key)

    class _Mouse:
        def __init__(self, page):
            self.page = page

        async def wheel(self, dx, dy):
            self.page.scrolled.append(dy)

    @property
    def keyboard(self):
        return self._Kbd(self)

    @property
    def mouse(self):
        return self._Mouse(self)


class TestExecuteAction:
    async def test_type_with_comma_in_selector(self):
        # Regression: selectors containing commas must not be split apart
        page = _FakePage()
        action = 'type("Explore websites, people, and locations", "scholarship")'
        done, msg = await _execute_action(page, action)
        assert not done
        assert page.filled == [("Explore websites, people, and locations", "scholarship")]

    async def test_type_single_quotes(self):
        page = _FakePage()
        done, _ = await _execute_action(page, "type('Search', 'gift')")
        assert not done
        assert page.filled == [("Search", "gift")]

    async def test_click_records_target(self):
        page = _FakePage()
        done, msg = await _execute_action(page, 'click("Sign In")')
        assert not done
        assert page.clicked == ["Sign In"]

    async def test_press_enter(self):
        page = _FakePage()
        done, _ = await _execute_action(page, "press_enter()")
        assert not done
        assert page.pressed == ["Enter"]

    async def test_scroll_direction(self):
        page = _FakePage()
        await _execute_action(page, "scroll('down')")
        await _execute_action(page, "scroll('up')")
        assert page.scrolled == [500, -500]

    async def test_done_terminates(self):
        page = _FakePage()
        done, msg = await _execute_action(page, "done(true, 'completed')")
        assert done

    async def test_unknown_action_feeds_back(self):
        page = _FakePage()
        done, msg = await _execute_action(page, "teleport('nowhere')")
        assert not done
        assert "Unknown action" in msg


# ---------------------------------------------------------------------------
# _verify_success — independent verification of the final page state
# ---------------------------------------------------------------------------

def _task(**kw):
    defaults = dict(
        url="https://www.example.com",
        description="Find the privacy policy",
        success_criteria="Privacy policy displayed",
        task_id="example.com::info",
        category="info",
        success_keywords=["privacy"],
    )
    defaults.update(kw)
    return WebTask(**defaults)


class TestVerifySuccess:
    async def test_verified_when_navigated_and_keyword_present(self):
        page = _FakePage(url="https://www.example.com/legal/privacy",
                         body_text="Our privacy policy explains ...")
        ok, final = await _verify_success(page, _task(),
                                          landed_url="https://www.example.com")
        assert ok

    async def test_not_verified_on_unchanged_homepage(self):
        # Homepage contains the keyword in nav text, but agent never navigated
        page = _FakePage(url="https://www.example.com",
                         body_text="Home | About | Privacy | Contact")
        ok, _ = await _verify_success(page, _task(),
                                      landed_url="https://www.example.com")
        assert not ok

    async def test_not_verified_when_only_redirect_happened(self):
        # Load-time redirect must not count as navigation (cnn -> edition.cnn)
        page = _FakePage(url="https://edition.example.com/",
                         body_text="Privacy notice here")
        ok, _ = await _verify_success(page, _task(url="https://www.example.com"),
                                      landed_url="https://edition.example.com/")
        assert not ok

    async def test_not_verified_without_keyword(self):
        page = _FakePage(url="https://www.example.com/somewhere",
                         body_text="entirely unrelated content")
        ok, _ = await _verify_success(page, _task(),
                                      landed_url="https://www.example.com")
        assert not ok


# ---------------------------------------------------------------------------
# Task generation — comparable tasks across the whole sample
# ---------------------------------------------------------------------------

def _site(sector="ecommerce", domain="shop.example"):
    return SiteRecord(domain=domain, name="Example", sector=sector,
                      url=f"https://www.{domain}", compliance_score=0.9,
                      compliance_bin="high")


class TestTaskTemplates:
    def test_three_tasks_per_site(self):
        tasks = tasks_for_site(_site())
        assert len(tasks) == 3
        assert {t.category for t in tasks} == {"navigation", "search", "info"}

    def test_every_sector_has_params(self):
        for sector in ["ecommerce", "education", "government", "healthcare", "news"]:
            assert sector in SECTOR_PARAMS
            tasks = tasks_for_site(_site(sector=sector))
            assert all(t.success_keywords for t in tasks)

    def test_task_ids_are_unique_and_stable(self):
        t1 = tasks_for_site(_site())
        t2 = tasks_for_site(_site())
        assert [t.task_id for t in t1] == [t.task_id for t in t2]
        assert len({t.task_id for t in t1}) == 3

    def test_generate_all_tasks_from_sample(self, tmp_path):
        import csv
        p = tmp_path / "sample.csv"
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "domain", "name", "sector", "url", "tranco_rank",
                "compliance_score", "compliance_bin"])
            w.writeheader()
            for i, s in enumerate(["ecommerce", "education", "government",
                                   "healthcare", "news"]):
                w.writerow({"domain": f"site{i}.example", "name": f"S{i}",
                            "sector": s, "url": f"https://site{i}.example",
                            "tranco_rank": i, "compliance_score": 0.8,
                            "compliance_bin": "medium"})
        pairs = generate_all_tasks(p)
        assert len(pairs) == 15  # 5 sites x 3 categories


# ---------------------------------------------------------------------------
# Bounded observation context
# ---------------------------------------------------------------------------

class TestStepMessages:
    def test_single_message_with_history_and_current_observation(self):
        obs = Observation(url="https://x.example", mode="accessibility_tree",
                          content="[RootWebArea] \"X\"", token_estimate=5)
        history = [("click('A')", "Clicked: A"), ("scroll(down)", "Scrolled down")]
        msgs = _build_step_messages(_task(), obs, history, step_num=3,
                                    observation_mode="accessibility_tree")
        assert len(msgs) == 1  # prior observations are never resent
        content = msgs[0]["content"]
        assert "Step 3" in content
        assert "click('A') -> Clicked: A" in content
        assert "[RootWebArea]" in content
