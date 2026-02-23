"""네이버 카페 게시글 상세 페이지 HTML 파서. 제목, 본문, 첨부 이미지, 댓글 추출.

저장 JSON 형식:
{
  "article_id": "44443",
  "title": "제목",
  "body": "본문 (질문 글 작성 전 필독 사항 ~ 조언도 함께 부탁드릴게요 구간)",
  "attachments": [ {"url": "https://...", "alt": ""}, ... ],
  "comments": [ {"author": "닉네임", "content": "댓글 본문"}, ... ],
  "written_date": "YYYY.MM.DD. HH:MM",
  "page_url": "https://...",
  "board": "게시판명(예: ◈펫보험)"
}

기대 HTML: 본문 마커(필독 사항/조언도 함께), img.se-image-resource, .comment_nickname, .text_comment
"""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

# 본문 시작/끝 마커 텍스트
_BODY_START_MARKER = "질문 글 작성 전 필독 사항"
_BODY_END_MARKER = "조언도 함께 부탁드릴게요"

# 글번호 추출
_ARTICLE_ID_RE = re.compile(r"/?articles?/(\d+)")


def _article_id_from_url(url: str) -> str | None:
    m = _ARTICLE_ID_RE.search(url)
    return m.group(1) if m else None


def _text_with_newlines(element) -> str:
    """요소의 텍스트를 <br>을 줄바꿈으로 변환하여 반환."""
    if element is None:
        return ""
    text = str(element)
    # <br>, <br/> → \n
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_body(soup: BeautifulSoup) -> str:
    """본문: '질문 글 작성 전 필독 사항' ~ '조언도 함께 부탁드릴게요' 구간 텍스트."""
    # 본문이 있는 컨테이너 우선 탐색
    for sel in (".app_content", ".content_body", "[class*='ArticleContent']", ".se-component-content", "article"):
        container = soup.select_one(sel)
        if not container:
            continue
        text = container.get_text(separator="\n", strip=True)
        if _BODY_START_MARKER in text and _BODY_END_MARKER in text:
            # 시작~끝 구간만 잘라냄
            start_i = text.find(_BODY_START_MARKER)
            end_i = text.find(_BODY_END_MARKER) + len(_BODY_END_MARKER)
            return text[start_i:end_i].strip()
        if text and len(text) > 100:
            return _text_with_newlines(container)
    # 마커가 있는 아무 요소에서 전체 텍스트 구한 뒤 구간 잘라냄
    el = soup.find(string=re.compile(re.escape(_BODY_START_MARKER)))
    if el:
        root = el.parent
        for _ in range(15):
            if root is None:
                break
            full = root.get_text(separator="\n", strip=True)
            if _BODY_END_MARKER in full:
                start_i = full.find(_BODY_START_MARKER)
                end_i = full.find(_BODY_END_MARKER) + len(_BODY_END_MARKER)
                return full[start_i:end_i].strip()
            root = getattr(root, "parent", None)
    return ""


def _get_body_container(soup: BeautifulSoup):
    """본문 구간(필독~조언도)을 포함하는 컨테이너 요소. 없으면 None."""
    el = soup.find(string=re.compile(re.escape(_BODY_START_MARKER)))
    if not el:
        return None
    root = el.parent
    for _ in range(20):
        if root is None:
            return None
        text = root.get_text(separator=" ", strip=True)
        if _BODY_END_MARKER in text:
            return root
        root = getattr(root, "parent", None)
    return None


def _extract_attachments(soup: BeautifulSoup) -> list[dict]:
    """첨부 이미지: 본문 구간(+부모 한 단계) 내 이미지만. 프로필/배너/썸네일 제외."""
    body_el = _get_body_container(soup)
    # 본문 마커를 포함하는 컨테이너 및 상위 2단계에서만 이미지 수집
    search_el = body_el
    imgs = []
    for _ in range(3):
        if search_el:
            imgs.extend(search_el.find_all("img", src=True))
            search_el = getattr(search_el, "parent", None)
    if not imgs:
        imgs = soup.find_all("img", src=True)
    images = []
    seen = set()
    # 제외할 URL 패턴 (프로필, 아바타, 배너, 버튼 등)
    skip_patterns = (
        "default_thumb", "web-section", "static/img", "c77_77", "f100_100",
        "profile", "avatar", "배너", "banner", "btn_layer", "editor/btn",
        "mask", "default_thumb.svg",
    )
    for img in imgs:
        src = img.get("src", "").strip()
        if not src or src in seen:
            continue
        if any(p in src.lower() for p in skip_patterns):
            continue
        if "cafeptthumb" not in src:
            continue
        seen.add(src)
        images.append({"url": src, "alt": (img.get("alt") or "").strip()})
    return images


_COMMENT_COUNT_RE = re.compile(r"댓글\s*(\d+)", re.IGNORECASE)


def _get_comment_count_from_page(soup: BeautifulSoup) -> int | None:
    """페이지에서 표시된 상위 댓글 수(예: '댓글 5')를 찾아 반환. 없으면 None."""
    text = soup.get_text()
    m = _COMMENT_COUNT_RE.search(text)
    if m:
        return int(m.group(1))
    return None


def _extract_comments(soup: BeautifulSoup) -> list[dict]:
    """댓글: 각 댓글 블록당 한 건. 페이지에 '댓글 N'이 있으면 상위 N개만, 없으면 전부."""
    expected_n = _get_comment_count_from_page(soup)
    nick_els = soup.find_all(class_=re.compile(r"comment_nickname|comment.*nick"))
    text_els = soup.find_all(class_=re.compile(r"text_comment"))
    if not nick_els or not text_els:
        return []

    # 페이지에 '댓글 N'이 있고, 닉네임/텍스트 개수가 N 이상이면 문서 순서로 1:1 매칭 후 상위 N개만 사용
    if expected_n is not None and expected_n >= 1 and len(nick_els) >= expected_n and len(text_els) >= expected_n:
        comments = []
        for i in range(expected_n):
            author = (nick_els[i].get_text(strip=True) or "").strip()
            content = _text_with_newlines(text_els[i]).strip()
            if author or content:
                comments.append({"author": author or "", "content": content})
        if comments:
            return comments

    # fallback: 같은 부모 기준으로 닉네임-텍스트 매칭
    comments = []
    used_text_ids = set()
    for nick_el in nick_els:
        author = (nick_el.get_text(strip=True) or "").strip()
        if not author:
            continue
        parent = nick_el.parent
        text_el = None
        for _ in range(15):
            if parent is None:
                break
            for t in text_els:
                if id(t) in used_text_ids:
                    continue
                p = getattr(t, "parent", None)
                for _ in range(20):
                    if p is None:
                        break
                    if p == parent:
                        text_el = t
                        break
                    p = getattr(p, "parent", None)
                if text_el:
                    break
            if text_el:
                break
            parent = getattr(parent, "parent", None)
        content = _text_with_newlines(text_el) if text_el else ""
        if text_el:
            used_text_ids.add(id(text_el))
        comments.append({"author": author, "content": content})
    if expected_n is not None and expected_n >= 1 and len(comments) > expected_n:
        comments = comments[:expected_n]
    return comments


# 작성 일자: YYYY.MM.DD. HH:MM (게시글 상단 노출 패턴)
_WRITTEN_DATE_RE = re.compile(r"\d{4}\.\d{2}\.\d{2}\.\s*\d{1,2}:\d{2}")


def _extract_board(soup: BeautifulSoup) -> str:
    """페이지에서 게시글이 속한 게시판 이름 추출. link_board 클래스의 링크 텍스트(ArticleList.nhn 등) 사용."""
    for a in soup.find_all("a", href=True, class_=re.compile(r"link_board")):
        href = a.get("href", "")
        if "ArticleList" in href or "menuid=" in href:
            name = a.get_text(strip=True)
            if name and len(name) <= 100:
                return name
    for a in soup.find_all("a", href=True):
        href = a.get("href", "")
        if "ArticleList.nhn" in href and "menuid=" in href:
            name = a.get_text(strip=True)
            if name and len(name) <= 100 and "목록" not in name:
                return name
    return ""


def _extract_written_date(soup: BeautifulSoup) -> str:
    """페이지에서 게시글 작성 일자(YYYY.MM.DD. HH:MM)를 추출. '조회' 앞 첫 번째 날짜를 사용."""
    text = soup.get_text(separator="\n", strip=True)
    # '조회' 이전 구간에서 첫 날짜를 사용(게시글 메타에 있는 작성일)
    if "조회" in text:
        head = text.split("조회")[0]
        m = _WRITTEN_DATE_RE.search(head)
        if m:
            return m.group(0).strip()
    m = _WRITTEN_DATE_RE.search(text)
    return m.group(0).strip() if m else ""


def _extract_title(soup: BeautifulSoup) -> str:
    """제목: 본문 제목 영역. 카페명(보험인스포유 등) 제외."""
    skip = ("보험인스포유", "보험비교", "상담", "네이버 카페", "전체글보기", "목록")
    for sel in (".tit_article", "[class*='ArticleTitle']", ".article_title", "h1", ".title", "h2", "[class*='tit']"):
        try:
            for el in soup.select(sel):
                t = el.get_text(strip=True)
                if not t or len(t) < 2 or len(t) > 400:
                    continue
                if any(s in t for s in skip) and len(t) < 80:
                    continue
                if "댓글" in t or "목록" in t:
                    continue
                return t
        except Exception:
            continue
    return ""


def parse_article_html(html: str, article_url: str = "") -> dict:
    """
    게시글 HTML에서 제목, 본문, 첨부 이미지, 댓글을 추출하여 dict로 반환한다.

    Returns:
        {
            "article_id": str,
            "title": str,
            "body": str,
            "attachments": [ {"url": str, "alt": str}, ... ],
            "comments": [ {"author": str, "content": str}, ... ],
            "written_date": str,
            "page_url": str,
            "board": str
        }
    """
    soup = BeautifulSoup(html, "html.parser")
    article_id = _article_id_from_url(article_url) if article_url else None
    if not article_id:
        for a in soup.find_all("a", href=True):
            aid = _article_id_from_url(a.get("href", ""))
            if aid:
                article_id = aid
                break

    return {
        "article_id": article_id or "",
        "title": _extract_title(soup),
        "body": _extract_body(soup),
        "attachments": _extract_attachments(soup),
        "comments": _extract_comments(soup),
        "written_date": _extract_written_date(soup),
        "page_url": article_url if article_url else "",
        "board": _extract_board(soup),
    }


def _get_article_frame(page):
    """게시글 본문이 로드된 프레임을 반환. cafe_main iframe 우선."""
    try:
        page.wait_for_selector("iframe#cafe_main, iframe[name='cafe_main'], iframe[src*='articles']", timeout=12000)
    except Exception:
        pass
    page.wait_for_timeout(4000)
    # 네이버 카페: 본문은 iframe cafe_main 안에 로드됨
    try:
        frame = page.frame(name="cafe_main")
        if frame:
            frame.wait_for_load_state("domcontentloaded", timeout=8000)
            page.wait_for_timeout(3000)
            return frame
    except Exception:
        pass
    try:
        frame = page.frame(id="cafe_main")
        if frame:
            return frame
    except Exception:
        pass
    best_frame = page.main_frame
    best_score = -1
    for frame in page.frames():
        try:
            content = frame.content()
            if _BODY_START_MARKER in content and _BODY_END_MARKER in content:
                return frame
            score = sum(1 for k in ("text_comment", "comment_nickname", "se-image-resource", "질문 글 작성", "필독", "조언도") if k in content)
            if score > best_score:
                best_score = score
                best_frame = frame
            if "text_comment" in content and len(content) > 10000:
                best_frame = frame
                best_score = max(best_score, 10)
        except Exception:
            continue
    return best_frame


def _get_article_page_html(page) -> str:
    """게시글 본문이 있는 프레임의 HTML 반환."""
    frame = _get_article_frame(page)
    return frame.content()


def _get_iframe_article_url(page) -> str | None:
    """메인 페이지에서 게시글 본문 iframe의 src(절대 URL)를 반환. cafe_main 우선."""
    from urllib.parse import urljoin
    try:
        iframe = page.query_selector("iframe#cafe_main, iframe[name='cafe_main']")
        if iframe:
            src = iframe.get_attribute("src")
            if src:
                return urljoin(page.url, src)
    except Exception:
        pass
    try:
        iframe = page.query_selector("iframe[src*='articles']")
        if iframe:
            src = iframe.get_attribute("src")
            if src:
                return urljoin(page.url, src)
    except Exception:
        pass
    for frame in page.frames():
        if frame != page.main_frame and frame.url != "about:blank":
            u = frame.url
            if "/articles/" in u or "ca-fe/cafes" in u:
                return u
    return None


def fetch_and_parse_article(
    url: str,
    timeout: int = 30,
    use_browser: bool = True,
) -> dict:
    """URL에서 게시글 HTML을 가져와 파싱한 결과 dict를 반환한다."""
    html = ""
    if use_browser:
        try:
            from playwright.sync_api import sync_playwright
            from urllib.parse import urljoin
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                page.wait_for_load_state("networkidle", timeout=20000)
                page.wait_for_timeout(5000)
                html = ""
                iframe_url = _get_iframe_article_url(page)
                if iframe_url:
                    try:
                        page.goto(iframe_url, wait_until="domcontentloaded", timeout=timeout * 1000)
                        page.wait_for_load_state("networkidle", timeout=15000)
                        page.wait_for_timeout(5000)
                        html = page.content()
                    except Exception:
                        pass
                if not html or ("필독" not in html and "text_comment" not in html):
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                    page.wait_for_timeout(3000)
                    html = _get_article_page_html(page)
                browser.close()
        except Exception:
            pass
    if not html:
        from app.customer_research.feature.http_client import fetch_html_safe
        got, _ = fetch_html_safe(url, timeout=timeout)
        html = got or ""
    data = parse_article_html(html, article_url=url)
    # 본문이 비어있을 때 디버깅용 HTML 저장 (SAVE_ARTICLE_DEBUG=1)
    try:
        import os
        if os.environ.get("SAVE_ARTICLE_DEBUG") and not data.get("body"):
            out_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "articles"
            out_dir.mkdir(parents=True, exist_ok=True)
            aid = data.get("article_id") or "unknown"
            (out_dir / f"{aid}_debug.html").write_text(html, encoding="utf-8")
    except Exception:
        pass
    return data


def save_article_json(data: dict, output_dir: str | Path) -> Path:
    """파싱 결과를 글번호.json으로 저장한다. output_dir이 없으면 생성."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aid = data.get("article_id") or "unknown"
    path = output_dir / f"{aid}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
