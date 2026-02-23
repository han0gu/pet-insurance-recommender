"""웹 페이지 HTTP 요청 공통 모듈."""

import requests
from requests.exceptions import RequestException


def fetch_html(url: str, timeout: int = 30, headers: dict | None = None) -> str:
    """
    URL에 GET 요청을 보내 응답 본문(HTML 등) 문자열을 반환한다.

    Args:
        url: 요청할 URL.
        timeout: 요청 타임아웃(초). 기본 30초.
        headers: 추가할 HTTP 헤더. None이면 기본 User-Agent만 사용.

    Returns:
        응답 body 텍스트(인코딩은 response.encoding 기준).

    Raises:
        RequestException: 네트워크 오류 또는 HTTP 4xx/5xx 시.
    """
    if headers is None:
        headers = _default_headers()
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    if response.encoding is None:
        response.encoding = "utf-8"
    return response.text


def fetch_html_safe(url: str, timeout: int = 30, headers: dict | None = None) -> tuple[str | None, str | None]:
    """
    URL에 GET 요청을 보내 응답 본문을 반환한다. 예외 시 None과 오류 메시지를 반환한다.

    Args:
        url: 요청할 URL.
        timeout: 요청 타임아웃(초).
        headers: 추가할 HTTP 헤더.

    Returns:
        (성공 시 HTML 문자열, 실패 시 None), (실패 시 오류 메시지, 성공 시 None).
    """
    try:
        html = fetch_html(url, timeout=timeout, headers=headers)
        return html, None
    except RequestException as e:
        return None, str(e)


def _default_headers() -> dict[str, str]:
    """기본 요청 헤더(브라우저처럼 보이도록)."""
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }
