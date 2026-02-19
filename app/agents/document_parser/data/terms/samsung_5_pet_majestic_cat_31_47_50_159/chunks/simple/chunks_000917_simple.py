from langchain_core.documents import Document

chunk = Document(
    page_content=('8) 약간의 운동장해 머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경추)를 제외한 척추체(척추뼈 몸 통)에 골절 또는 탈구 등으로 '
 '2개의 척추체(척추뼈 몸통)를 유합(아물어 붙음) 또는 고정한 상태 9) 심한 기형이란 다음 중 어느 하나에 해당하는 경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 141},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000917',
              'chunk_char_len': 148,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
