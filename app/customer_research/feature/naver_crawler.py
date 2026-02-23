"""네이버 카페 등 네이버 웹페이지 HTML 수집 모듈."""

import re
from urllib.parse import urlencode, urlparse, urlunparse

from app.customer_research.feature.http_client import fetch_html, fetch_html_safe
from app.customer_research.feature.naver_cafe_parser import get_article_titles_from_html, get_next_page_url


def _build_next_page_url(base_url: str, page_num: int) -> str:
    """URL에 페이지 번호 파라미터를 붙인 다음 페이지 URL을 만든다. (HTML에서 다음 링크를 못 찾을 때 사용)"""
    parsed = urlparse(base_url)
    query = parsed.query
    params = {}
    if query:
        for part in query.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                params[k] = v
    params["searchPageNum"] = str(page_num)
    new_query = urlencode(params)
    return urlunparse(parsed._replace(query=new_query))


def fetch_naver_cafe_menu(url: str, timeout: int = 30) -> str:
    """네이버 카페 메뉴(게시판) 페이지의 HTML 소스를 가져온다."""
    return fetch_html(url, timeout=timeout)


def fetch_naver_cafe_menu_safe(url: str, timeout: int = 30) -> tuple[str | None, str | None]:
    """네이버 카페 메뉴 페이지 HTML을 가져온다. 실패 시 (None, 오류메시지) 반환."""
    return fetch_html_safe(url, timeout=timeout)


def fetch_naver_page(url: str, timeout: int = 30) -> str:
    """네이버 도메인 페이지 HTML을 가져온다."""
    return fetch_html(url, timeout=timeout)


def _get_content_frame(page):
    """목록+페이지네이션이 있는 프레임을 반환한다. 메인에 있으면 main_frame."""
    main_html = page.content()
    if "/articles/" in main_html or "articles/" in main_html:
        return page.main_frame
    for frame in page.frames():
        if frame == page.main_frame:
            continue
        try:
            content = frame.content()
            if content and ("/articles/" in content or "articles/" in content):
                return frame
        except Exception:
            continue
    return page.main_frame


def _get_page_html(page) -> tuple[str, str]:
    """Playwright page에서 본문 HTML과 base_url을 반환한다. 메인에 목록이 없으면 iframe 내용을 시도한다."""
    frame = _get_content_frame(page)
    html = frame.content()
    url = frame.url if frame != page.main_frame else page.url
    return html, url


def _fetch_html_with_browser(url: str, timeout: int = 30) -> str | None:
    """Playwright로 페이지를 렌더링한 뒤 HTML을 반환한다. 실패 시 None."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return None
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            page.wait_for_timeout(3000)
            html, _ = _get_page_html(page)
            browser.close()
            return html
    except Exception:
        return None


def get_cafe_menu_article_titles(
    url: str,
    timeout: int = 30,
    use_browser: bool = True,
    max_pages: int | None = None,
) -> list[tuple[str | None, str, str]]:
    """네이버 카페 게시판 URL에서 (글번호, 제목, 게시글 링크) 목록을 반환한다. 다음 페이지가 있으면 계속 수집한다.
    max_pages: 수집할 최대 페이지 수. None이면 다음이 있는 한 계속 수집."""
    all_items: list[tuple[str | None, str, str]] = []
    current_url: str | None = url
    page_count = 0

    def _fetch(u: str) -> str:
        if use_browser:
            h = _fetch_html_with_browser(u, timeout=timeout)
            if h is not None:
                return h
        got, _ = fetch_naver_cafe_menu_safe(u, timeout=timeout)
        return got or ""

    if use_browser:
        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                try:
                    page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
                    page.wait_for_timeout(3000)
                    frame = _get_content_frame(page)
                    base = frame.url if frame != page.main_frame else page.url

                    while max_pages is None or page_count < max_pages:
                        html = frame.content()
                        items = get_article_titles_from_html(html, base_url=base)
                        if not items and page_count >= 1:
                            break
                        seen_links = {link for _, _, link in all_items}
                        new_items = [(aid, t, link) for aid, t, link in items if link not in seen_links]
                        if not new_items and page_count >= 1:
                            break
                        all_items.extend(new_items)
                        page_count += 1
                        if max_pages is not None and page_count >= max_pages:
                            break
                        # 한 페이지씩 이동: 현재 페이지(aria-current="page") 다음 숫자 버튼 클릭. 없으면 "다음" 클릭
                        try:
                            cur_btn = frame.locator("button.btn.number[aria-current='page']").first
                            if cur_btn.count() == 0:
                                next_btn = frame.locator("button[aria-label='다음'], button.type_next").first
                                if next_btn.count() == 0 or next_btn.is_disabled():
                                    break
                                next_btn.click()
                            else:
                                cur_num = int((cur_btn.text_content() or "1").strip())
                                next_num = cur_num + 1
                                next_num_btn = frame.locator("button.btn.number").filter(has_text=re.compile(f"^{next_num}$"))
                                if next_num_btn.count() > 0:
                                    next_num_btn.first.click()
                                else:
                                    next_btn = frame.locator("button[aria-label='다음'], button.type_next").first
                                    if next_btn.count() == 0 or next_btn.is_disabled():
                                        break
                                    next_btn.click()
                            page.wait_for_timeout(2000)
                        except Exception:
                            break
                finally:
                    browser.close()
            return all_items
        except Exception:
            use_browser = False
            current_url = url
            all_items = []
            page_count = 0

    while current_url and (max_pages is None or page_count < max_pages):
        html = _fetch(current_url)
        items = get_article_titles_from_html(html, base_url=current_url)
        if not items and page_count >= 1:
            break
        seen_links = {link for _, _, link in all_items}
        new_items = [(aid, t, link) for aid, t, link in items if link not in seen_links]
        if not new_items and page_count >= 1:
            break
        all_items.extend(new_items)
        page_count += 1
        next_url = get_next_page_url(html, base_url=current_url)
        if next_url is None and items:
            next_url = _build_next_page_url(current_url, page_count + 1)
        current_url = next_url
    return all_items
