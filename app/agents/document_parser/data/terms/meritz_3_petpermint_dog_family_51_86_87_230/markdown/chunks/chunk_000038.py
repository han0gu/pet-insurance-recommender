from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 2025년 4월 1일 | 1천만원 +1천만원 ÷ (1 + 평균공시이율) +1천만원 ÷ (1 + 평균공시이율)2 |\n'
 '| 2026년 4월 1일 | - |\n'
 '| 2027년 4월 1일 | - |\n'
 '# 제12조(주소변경통지)\uf000 계약자(보험수익자가 계약자와 다른 경우 보험수익자를\n'
 '포함합니다)는 주소 또는 연락처가 변경된 경우에는 지체없\n'
 '이 그 변경내용을 회사에 알려야 합니다.\n'
 '\uf000 제1항에서 정한대로 계약자 또는 보험수익자가 변경내용\n'
 '을 알리지 않은 경우에는 계약자 또는 보험수익자가 회사에'),
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
