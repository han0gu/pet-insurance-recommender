from langchain_core.documents import Document

chunk = Document(
    page_content=('⑤ 제3항에 의한 계약의 해지는 손해가 생긴 후에 이루어진 경우에도 회사는 그 손해를 보상하여 드 리지 않습니다. 손해가 제3항 제1호 '
 '및 제2호의 사실로 생긴 것이 아님을 계약자 또는 피보험자가 증명한 경우에는 보상하여 드립니다. ⑥ 회사는 다른 보험가입내역에 대한 계약 '
 '전·후 알릴 의무 위반을 이유로 계약을 해지하거나 보험금 지급을 거절하지 않습니다.\n'
 '제27조(중대사유로 인한 해지)\n'
 '① 회사는 아래와 같은 사실이 있을 경우에는 그 사실을 안 날부터 1개월 이내에 계약을 해지할 수 있 습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 16},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000085',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
