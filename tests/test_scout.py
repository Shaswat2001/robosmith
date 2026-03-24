"""Tests for robosmith.stages.scout — literature search."""

from unittest.mock import patch, MagicMock

import pytest

from robosmith.config import RobotType, EnvironmentType, TaskSpec
from robosmith.stages.scout import (
    KnowledgeCard,
    build_search_queries,
    search_papers,
    run_scout,
)


# ── Mock API response ──

MOCK_S2_RESPONSE = {
    "total": 42,
    "data": [
        {
            "title": "Eureka: Human-Level Reward Design via Coding LLMs",
            "year": 2024,
            "citationCount": 150,
            "abstract": "We present Eureka, a reward design algorithm powered by LLMs.",
            "url": "https://semanticscholar.org/paper/abc123",
            "externalIds": {"ArXiv": "2310.12931"},
            "authors": [{"name": "Yecheng Jason Ma"}, {"name": "William Liang"}],
        },
        {
            "title": "Language to Rewards for Robotic Skill Synthesis",
            "year": 2023,
            "citationCount": 85,
            "abstract": "We introduce a paradigm using LLMs to define reward parameters.",
            "url": "https://semanticscholar.org/paper/def456",
            "externalIds": {"ArXiv": "2306.08647"},
            "authors": [{"name": "Wenhao Yu"}],
        },
        {
            "title": "Some Obscure Paper",
            "year": 2022,
            "citationCount": 2,
            "abstract": None,
            "url": "",
            "externalIds": None,
            "authors": [],
        },
    ],
}


def _mock_s2_get(*args, **kwargs):
    """Mock httpx response."""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = MOCK_S2_RESPONSE
    resp.raise_for_status = MagicMock()
    return resp


# ── KnowledgeCard ──


class TestKnowledgeCard:
    def test_top_papers_sorted(self):
        card = KnowledgeCard(
            query="test",
            papers=[
                {"title": "A", "citations": 10},
                {"title": "B", "citations": 100},
                {"title": "C", "citations": 50},
            ],
        )
        top = card.top_papers(2)
        assert top[0]["title"] == "B"
        assert top[1]["title"] == "C"

    def test_summary_no_papers(self):
        card = KnowledgeCard(query="nothing")
        assert "No papers found" in card.summary()

    def test_summary_with_papers(self):
        card = KnowledgeCard(
            query="test",
            papers=[{"title": "Cool Paper", "year": 2024, "citations": 42}],
        )
        s = card.summary()
        assert "Cool Paper" in s
        assert "42" in s


# ── Query building ──


class TestBuildSearchQueries:
    def test_locomotion_task(self):
        spec = TaskSpec(
            task_description="Walk forward fast",
            robot_type=RobotType.QUADRUPED,
            robot_model="unitree_go2",
        )
        queries = build_search_queries(spec)
        assert len(queries) >= 2
        assert any("reinforcement learning" in q for q in queries)
        assert any("locomotion" in q or "reward" in q for q in queries)

    def test_manipulation_task(self):
        spec = TaskSpec(
            task_description="Pick up a red cube",
            robot_type=RobotType.ARM,
            robot_model="franka",
        )
        queries = build_search_queries(spec)
        assert any("manipulation" in q or "reward" in q for q in queries)

    def test_pendulum_task(self):
        spec = TaskSpec(
            task_description="Swing up and balance the pendulum",
            robot_type=RobotType.CUSTOM,
        )
        queries = build_search_queries(spec)
        assert any("swing" in q or "balance" in q for q in queries)


# ── Paper search (mocked API) ──


class TestSearchPapers:
    @patch("robosmith.stages.scout.search.httpx.Client")
    def test_returns_papers(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_s2_get()
        mock_client_cls.return_value = mock_client

        card = search_papers("reward function design")

        assert len(card.papers) == 3
        assert card.total_found == 42
        assert card.papers[0]["title"] == "Eureka: Human-Level Reward Design via Coding LLMs"
        assert card.papers[0]["citations"] == 150
        assert card.papers[0]["arxiv_id"] == "2310.12931"

    @patch("robosmith.stages.scout.search.httpx.Client")
    def test_handles_none_fields(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.return_value = _mock_s2_get()
        mock_client_cls.return_value = mock_client

        card = search_papers("test")

        # Paper 3 has None abstract and externalIds
        obscure = card.papers[2]
        assert obscure["abstract"] == ""
        assert obscure["arxiv_id"] is None

    @patch("robosmith.stages.scout.search.httpx.Client")
    def test_api_error_returns_empty(self, mock_client_cls):
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.get.side_effect = Exception("Network error")
        mock_client_cls.return_value = mock_client

        card = search_papers("test")

        assert len(card.papers) == 0


# ── Full scout (mocked) ──


class TestRunScout:
    @patch("robosmith.stages.scout.search_papers")
    def test_merges_results(self, mock_search):
        # Two queries return overlapping papers
        mock_search.side_effect = [
            KnowledgeCard(query="q1", papers=[
                {"title": "Paper A", "citations": 100},
                {"title": "Paper B", "citations": 50},
            ], total_found=2),
            KnowledgeCard(query="q2", papers=[
                {"title": "Paper B", "citations": 50},  # duplicate
                {"title": "Paper C", "citations": 75},
            ], total_found=2),
            KnowledgeCard(query="q3", papers=[], total_found=0),
        ]

        spec = TaskSpec(
            task_description="Walk forward fast",
            robot_type=RobotType.QUADRUPED,
        )
        card = run_scout(spec)

        # Should deduplicate Paper B
        assert len(card.papers) == 3
        # Should be sorted by citations
        assert card.papers[0]["title"] == "Paper A"
        assert card.papers[1]["title"] == "Paper C"
        assert card.papers[2]["title"] == "Paper B"


class TestBuildLiteratureContext:
    def test_empty_card(self):
        from robosmith.stages.scout import build_literature_context
        card = KnowledgeCard(query="test")
        assert build_literature_context(card) == ""

    def test_produces_numbered_list(self):
        from robosmith.stages.scout import build_literature_context
        card = KnowledgeCard(
            query="test",
            papers=[
                {"title": "Eureka: Reward Design", "year": 2024, "citations": 150,
                 "abstract": "We present a method for automated reward design using LLMs."},
                {"title": "Language to Rewards", "year": 2023, "citations": 85,
                 "abstract": "A paradigm for defining reward parameters from language."},
            ],
        )
        ctx = build_literature_context(card)
        assert "1." in ctx
        assert "2." in ctx
        assert "Eureka" in ctx
        assert "Language to Rewards" in ctx
        assert "Key insight" in ctx

    def test_respects_max_papers(self):
        from robosmith.stages.scout import build_literature_context
        card = KnowledgeCard(
            query="test",
            papers=[{"title": f"Paper {i}", "citations": 100 - i, "abstract": ""}
                    for i in range(20)],
        )
        ctx = build_literature_context(card, max_papers=3)
        assert "3." in ctx
        assert "4." not in ctx
