from langchain_core.documents import Document

chunk = Document(
    page_content=('전환을 원할 경우 수익자 지정이 필요합니다.- ② 전환대상계약이 해지 또는 기타 사유로 효력이 없게 된 경우 또는 전환대상계약이 '
 '제1항에서 정한\n'
 '- 조건을 만족하지 않게 된 경우 이 특약은 그 때부터 효력이 없습니다.\n'
 '- ③ 제2조 제1항에 따라 제출된 장애인증명서상 장애예상기간(또는 장애기간)이 종료된 경우에는 제3\n'
 '- 조 제1항에도 불구하고 이 특약은 그때부터 효력이 없습니다.\n'
 '- ④ 이 특약의 계약자는 전환대상계약의 계약자와 동일하여야 합니다.'),
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
