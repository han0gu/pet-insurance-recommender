from langchain_core.documents import Document

chunk = Document(
    page_content=('. 4. 가산이율 적용시 「보험금의 지급절차」 제2항 각 호의 어느 하나에 해당되는 사유로 지연된 경우에는 해당기간에 대하여 가산이율을 '
 '적용하지 않습니다.(다만, 상해 · 질병 관련 보장에 한합니다) 5. 가산이율 적용시 금융위원회 또는 금융감독원이 정당한 사유로 인정하는 '
 '경우에는 해당 기간 에 대하여 가산이율을 적용하지 않습니다. 6'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 135},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000860',
              'chunk_char_len': 188,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
