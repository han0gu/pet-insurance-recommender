from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 증가된 위험과 관계없이 발생한 보험금 지급사유에 관해 서는 원래대로 지급합니다. ⑤ 계약자 또는 피보험자가 고의 또는 중대한 '
 '과실로 제1항 각 호의 변경사실을 회사에 알리지 않았을 경우 변경후 요율이 변경전 요율보다 높을 때에는 회사는 그 변경사실 을 안 날부터 '
 '1개월 이내에 계약자 또는 피보험자에게 제4항에 따라 보장됨을 통보하 고 이에 따라 보험금을 지급합니다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 101},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000582',
              'chunk_char_len': 211,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
