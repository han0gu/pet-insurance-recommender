"""네이버 카페 HTML에서 게시글 제목과 링크를 추출하는 파서."""

import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

# 제목으로 보기 어려운 링크 텍스트(필터링)
_SKIP_TEXTS = frozenset({
    "공지", "더보기", "이전", "다음", "목록", "검색", "글쓰기",
    "전체글보기", "이미지모아보기", "카페태그", "카페 캘린더",
    "로딩중", "로딩 중", ""
})

# 게시글 링크로 볼 수 있는 href 패턴
_ARTICLE_HREF_PATTERNS = ("/articles/", "articles/", "/article/")

# URL에서 글번호(article id) 추출용
_ARTICLE_ID_RE = re.compile(r"/?articles?/(\d+)")

# 다음 페이지 링크로 볼 수 있는 텍스트/속성
_NEXT_PAGE_TEXT = ("다음", "다음 페이지", "다음페이지", ">", "next", "Next")
_NEXT_PAGE_REL = "next"
_NEXT_PAGE_CLASS_KEYWORDS = ("next", "pg_next", "btn_next", "page_next", "pagination_next")

# 댓글 수 표기 제거용
_COMMENT_PATTERNS = re.compile(
    r"(?:(?:\s*[·]\s*)?댓글\s*\d+(?:개)?|\s*\(\d+\)|\s*댓글수\s*\[\d+\])\s*$"
)

# 공지글 제목 접두사
_NOTICE_PREFIXES = (
    "[공지]", "[공지] ", "【공지】", "【공지】 ", "(공지)", "(공지) ",
    "공지]", "공지] ", "공지:", "공지 :",
)

# 공지 관련 DOM
_NOTICE_CLASS_KEYWORDS = ("notice", "공지", "notify", "pin", "pinned", "fixed", "sticky")
_NOTICE_ATTR_KEYS = ("data-article-type", "data-type", "data-notice", "data-is-notice", "data-pinned")
_NOTICE_ATTR_VALUES = ("notice", "notify", "true", "1")


def _is_notice_element(element) -> bool:
    """요소 또는 부모에 공지 관련 class/data 또는 '공지' 텍스트가 있으면 True."""
    if element is None:
        return False
    current, checked, max_levels = element, 0, 15
    while current and checked < max_levels:
        checked += 1
        for cls in current.get("class") or []:
            if isinstance(cls, str) and any(kw in cls.lower() or kw in cls for kw in _NOTICE_CLASS_KEYWORDS):
                return True
        for key, val in (current.attrs or {}).items():
            key_lower = key.lower()
            val_str = (str(val)).lower() if val else ""
            if "notice" in key_lower or "pin" in key_lower:
                if not val or val_str in _NOTICE_ATTR_VALUES:
                    return True
            if key in _NOTICE_ATTR_KEYS and val_str in _NOTICE_ATTR_VALUES:
                return True
        parent = getattr(current, "parent", None)
        if parent:
            for child in parent.children:
                if child is current:
                    continue
                text = child.strip() if getattr(child, "strip", None) and isinstance(child, str) else (child.get_text(strip=True) if getattr(child, "get_text", None) else "")
                if text == "공지":
                    return True
        current = getattr(current, "parent", None)
    return False


def _is_notice_title(title: str) -> bool:
    t = title.strip()
    return any(t.startswith(p) for p in _NOTICE_PREFIXES) or t == "공지"


def _strip_comment_suffix(text: str) -> str:
    return _COMMENT_PATTERNS.sub("", text).strip()


def _article_id_from_href(href: str) -> str | None:
    """게시글 URL(href)에서 글번호를 추출한다. 예: /articles/64944 -> '64944'."""
    m = _ARTICLE_ID_RE.search(href)
    return m.group(1) if m else None


def get_article_titles_from_html(
    html: str,
    base_url: str = "",
    min_title_length: int = 2,
) -> list[tuple[str | None, str, str]]:
    """네이버 카페 게시판 목록 HTML에서 (글번호, 제목, 게시글 링크) 목록을 추출한다. 공지글은 제외한다. 글번호는 URL에서 추출하며 없으면 None."""
    soup = BeautifulSoup(html, "html.parser")
    seen, result = set(), []

    def add(title: str, href: str) -> None:
        t = _strip_comment_suffix(title.strip())
        if len(t) < min_title_length or t in _SKIP_TEXTS or _is_notice_title(t):
            return
        link = urljoin(base_url, href) if base_url else href
        if link in seen:
            return
        seen.add(link)
        article_id = _article_id_from_href(href) or _article_id_from_href(link)
        result.append((article_id, t, link))

    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if any(p in href for p in _ARTICLE_HREF_PATTERNS):
            if _is_notice_element(a):
                continue
            add(a.get_text(strip=True), href)

    for item in soup.find_all(attrs={"role": "listitem"}) or soup.select("[data-list-type='article']"):
        if _is_notice_element(item):
            continue
        a = item.find("a", href=True)
        if a:
            href = a.get("href", "")
            if any(p in href for p in _ARTICLE_HREF_PATTERNS):
                add(a.get_text(strip=True), href)

    return result


def _normalize_next_text(s: str) -> str:
    """다음 페이지 관련 텍스트 정규화."""
    t = s.strip().replace(" ", "").lower()
    if t in ("다음", "다음페이지", "next"):
        return t
    return s.strip()


def get_next_page_url(html: str, base_url: str = "") -> str | None:
    """목록 HTML에서 '다음 페이지' 링크 URL을 추출한다. 없으면 None."""
    soup = BeautifulSoup(html, "html.parser")

    def is_article_link(href: str) -> bool:
        return any(p in href for p in _ARTICLE_HREF_PATTERNS)

    def accept_href(href: str) -> str | None:
        href = href.strip()
        if not href or is_article_link(href):
            return None
        return urljoin(base_url, href)

    # rel="next"
    a = soup.find("a", attrs={"rel": _NEXT_PAGE_REL}, href=True)
    if a:
        u = accept_href(a.get("href", ""))
        if u:
            return u

    # aria-label / title에 '다음' 포함
    for a in soup.find_all("a", href=True):
        aria = (a.get("aria-label") or "").strip()
        title = (a.get("title") or "").strip()
        if "다음" in aria or "다음" in title or "next" in aria.lower() or "next" in title.lower():
            u = accept_href(a.get("href", ""))
            if u:
                return u

    # class에 next 관련 키워드
    for a in soup.find_all("a", href=True):
        for cls in a.get("class") or []:
            if isinstance(cls, str) and any(kw in cls.lower() for kw in _NEXT_PAGE_CLASS_KEYWORDS):
                u = accept_href(a.get("href", ""))
                if u:
                    return u

    # 텍스트가 '다음' 등인 링크 (게시글 링크가 아닌 것만)
    for a in soup.find_all("a", href=True):
        if is_article_link(a.get("href", "")):
            continue
        text = a.get_text(strip=True)
        if text in _NEXT_PAGE_TEXT:
            u = accept_href(a.get("href", ""))
            if u:
                return u
        # 공백 제거 후 비교
        normalized = _normalize_next_text(text)
        if normalized in ("다음", "다음페이지", "next"):
            u = accept_href(a.get("href", ""))
            if u:
                return u

    return None
