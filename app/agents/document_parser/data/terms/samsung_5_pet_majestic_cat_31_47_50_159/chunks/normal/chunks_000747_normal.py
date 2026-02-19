from langchain_core.documents import Document

chunk = Document(
    page_content=('. 반려묘 의료비(치과및구강질환포함)(재가입형) 특별약관 제27조 (특별약관의 재 가입에 관한 사항) 제5항에 따라 보험계약이 연장된 '
 '경우에는 종전 계약의 보험기간 을 연장하는 것으로 보아제2항을 적용하지 않습니다. ⑧ 제3항에도 불구하고 4-1. 반려묘 '
 '의료비(치과및구강질환포함)(재가입형) 특별약관 제 27조 (특별약관의 재가입에 관한 사항) 제1항 및 제2항에 따라 재가입하는 경우 또는 '
 '4-1'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 117},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000747',
              'chunk_char_len': 223,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
