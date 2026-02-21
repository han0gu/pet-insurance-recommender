from langchain_core.documents import Document

chunk = Document(
    page_content=('후, 보험증권에 기재된 보상비율(70%)을 곱한 금액을 아래\n'
 '에서 정한 금액을 한도로 보상합니다.| 항목 | 항목 | 항목 | 항목 | 자기 부담금 | 지급 한도 |\n'
 '| --- | --- | --- | --- | --- | --- |\n'
 '| 입 원 의 료 비 Ⅲ | 입원 중 수술을 하지 않은 날의 경우 | MRI,CT 및 내시경처치 를 받은 날의 경우 | 연간 첫번째 | '
 '1일당 3만원/ 5만원 중 보험증 권에 기재된 자기부 담금 | 1일당 50만원 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
