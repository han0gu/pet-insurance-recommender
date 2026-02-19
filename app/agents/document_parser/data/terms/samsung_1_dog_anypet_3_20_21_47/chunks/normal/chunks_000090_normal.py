from langchain_core.documents import Document

chunk = Document(
    page_content=('제27조(중대사유로 인한 해지)\n'
 '① 회사는 아래와 같은 사실이 있을 경우에는 그 사실을 안 날부터 1개월 이내에 계약을 해지할 수 있 습니다.\n'
 '1. 계약자 또는 피보험자가 보험금을 지급받을 목적으로 고의로 보험금 지급사유를 발생시킨 경우 2. 계약자 또는 피보험자가 보험금 청구에 '
 '관한 서류에 고의로 사실과 다른 것을 기재하였거나 그 서류 또는 증거를 위조 또는 변조한 경우. 다만, 이미 보험금 지급사유가 발생한 '
 '경우에는 보험 금 지급에 영향을 미치지 않습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000090',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
