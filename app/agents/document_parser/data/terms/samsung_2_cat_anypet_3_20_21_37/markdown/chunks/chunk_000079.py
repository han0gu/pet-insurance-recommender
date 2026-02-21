from langchain_core.documents import Document

chunk = Document(
    page_content=('의 전액, 효력상실 또는 해지의 경우에는 경과하지 않은 기간에 대하여 일단위로 계산한 보험료- 17 -당신에게 좋은보험 삼성화재2. '
 '계약자 또는 피보험자의 책임 있는 사유에 의하는 경우 : 이미 경과한 기간에 대하여 단기요율\n'
 '로 계산한 보험료를 뺀 잔액. 다만, 계약자, 피보험자의 고의 또는 중대한 과실로 무효가 된 때\n'
 '에는 보험료를 돌려드리지 않습니다.- ② 보험기간이 1년을 초과하는 계약이 무효 또는 효력상실인 경우에는 무효 또는 효력상실의 원인이'),
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
