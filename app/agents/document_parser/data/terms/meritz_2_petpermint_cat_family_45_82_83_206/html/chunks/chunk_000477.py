from langchain_core.documents import Document

chunk = Document(
    page_content=("아래<br>에서 정한 금액을 한도로 보상합니다.</p><br><table id='74' "
 "style='font-size:16px'><thead><tr><td "
 'colspan="2">항목</td><td>자기부담금</td><td>지급 한도</td></tr></thead><tbody><tr><td '
 'rowspan="2">입원 의료비</td><td>입원 중 수술을 하지 않은 날의 경우</td><td rowspan="2">1일당 '
 '3만원/5만원 중 보험증권에 기재된 자기부담금</td><td>1일당 30만원</td></tr><tr><td>입원 중'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
