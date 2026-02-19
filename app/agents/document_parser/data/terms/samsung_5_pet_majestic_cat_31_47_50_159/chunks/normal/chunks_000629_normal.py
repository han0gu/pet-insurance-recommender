from langchain_core.documents import Document

chunk = Document(
    page_content=('제25조 (중대사유로 인한 해지)\n'
 '① 회사는 아래와 같은 사실이 있을 경우에는 그 사실을 안 날부터 1개월 이내에 이 특별 약관을 해지할 수 있습니다.\n'
 '1. 계약자 또는 피보험자가 보험금을 지급받을 목적으로 고의로 보험금 지급사유를 발생시킨 경우\n'
 '2. 계약자 또는 피보험자가 보험금 청구에 관한 서류에 고의로 사실과 다른 것을 기재 하였거나 그 서류 또는 증거를 위조 또는 변조한 '
 '경우. 다만, 이미 보험금 지급사 유가 발생한 경우에는 보험금 지급에 영향을 미치지 않습니다.\n'
 '<용어풀이>'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 105},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000629',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
