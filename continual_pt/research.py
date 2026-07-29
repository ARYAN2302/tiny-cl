"""Web research with provenance and an explicit untrusted-content boundary."""
from __future__ import annotations

import re
from dataclasses import asdict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .schema import LearningGoal, SourceDocument


class WebResearcher:
    """Fetches reference material; fetched text is data, never executable instruction."""

    def __init__(self, timeout_seconds: int = 20, max_documents: int = 6, max_chars: int = 18_000):
        self.timeout_seconds = timeout_seconds
        self.max_documents = max_documents
        self.max_chars = max_chars

    def collect(self, goal: LearningGoal, queries: list[str] | None = None) -> list[SourceDocument]:
        # Preserve breadth across the learner's questions.  Flattening search
        # results makes the first query consume the entire document budget,
        # which is exactly how a multi-topic goal silently became an FSDP-only
        # curriculum in the first run.
        urls = list(dict.fromkeys(goal.sources + self._search(queries or goal.research_queries)))[: self.max_documents]
        documents = []
        for url in urls:
            if not self._allowed(url, goal.trusted_domains):
                continue
            document = self._fetch(url)
            if document:
                documents.append(document)
        return documents

    def _search(self, queries: list[str]) -> list[str]:
        if not queries:
            return []
        try:
            from ddgs import DDGS
            per_query: list[list[str]] = []
            with DDGS() as search:
                for query in queries:
                    print(f"[research] tool.search({query!r})", flush=True)
                    hits = []
                    for result in search.text(query, max_results=3):
                        href = result.get("href") or result.get("url")
                        if href:
                            hits.append(href)
                    per_query.append(hits)
            # Round-robin selection ensures the initial research set covers
            # every planned query before collecting second-choice results.
            urls = []
            for rank in range(max((len(hits) for hits in per_query), default=0)):
                for hits in per_query:
                    if rank < len(hits):
                        urls.append(hits[rank])
            return urls
        except Exception:
            return []

    def _allowed(self, url: str, trusted_domains: list[str]) -> bool:
        if urlparse(url).scheme not in {"http", "https"}:
            return False
        if not trusted_domains:
            return True
        host = (urlparse(url).hostname or "").lower()
        return any(host == domain or host.endswith("." + domain) for domain in trusted_domains)

    def _fetch(self, url: str) -> SourceDocument | None:
        try:
            response = requests.get(
                url,
                timeout=self.timeout_seconds,
                headers={"User-Agent": "continual-pt research bot/0.1"},
            )
            response.raise_for_status()
        except requests.RequestException:
            return None
        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer"]):
            tag.decompose()
        title = soup.title.get_text(" ", strip=True) if soup.title else url
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))[: self.max_chars]
        sections = self._sections(soup)
        return SourceDocument(url=url, title=title, text=text, sections=sections) if text else None

    @staticmethod
    def _sections(soup: BeautifulSoup) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        for heading in soup.find_all(["h1", "h2", "h3"]):
            title = heading.get_text(" ", strip=True)
            fragments = []
            for sibling in heading.find_all_next():
                if sibling is not heading and sibling.name in {"h1", "h2", "h3"}:
                    break
                if sibling.name in {"p", "li", "pre"}:
                    value = sibling.get_text(" ", strip=True)
                    if value:
                        fragments.append(value)
                if len(" ".join(fragments)) >= 900:
                    break
            body = re.sub(r"\s+", " ", " ".join(fragments))[:1_200]
            if len(title) >= 3 and len(body) >= 80:
                sections.append((title, body))
        return sections[:20]


def source_records(documents: list[SourceDocument]) -> list[dict]:
    return [asdict(document) for document in documents]
