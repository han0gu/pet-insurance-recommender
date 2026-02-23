from langchain_core.documents import Document

chunk = Document(
    page_content=('- 기간 만료일)까지 보상하여 드립니다.\n'
 '# 제5조(보험료의 환급)계약자의 책임있는 사유로 계약을 해지하는 경우에는 보통약관 제30조(보험료의 환급)의 규정에도 불'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
