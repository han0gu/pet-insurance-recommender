from langchain_core.documents import Document

chunk = Document(
    page_content=('. 한편<br>위험이 증가된 경우에는 보험료의 증액 및 정산금액의 추가<br>납입을 요구할 수 있으며, 계약자는 일시납 또는 잔여 '
 '보험<br>료 납입기간과 5년 중 큰 기간(단, 잔여 보험기간을 초과할<br>수 없음) 동안의 분납 중 선택하여 정산금액을 '
 '납입하여야<br>합니다'),
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
