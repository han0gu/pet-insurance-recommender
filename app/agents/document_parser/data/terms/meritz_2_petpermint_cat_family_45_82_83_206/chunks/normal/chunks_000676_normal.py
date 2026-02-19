from langchain_core.documents import Document

chunk = Document(
    page_content=('8) 약간의 운동장해 머리뼈(두개골)와 상위목뼈(상위경추: 제1, 2경추)를 제외한 척추체(척추뼈 몸통)에 골절 또는 탈구로 2개 의 '
 '척추체(척추뼈 몸통)를 유합(아물어 붙음) 또는 고 정한 상태 9) 심한 기형이란 다음 중 어느 하나에 해당하는 경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 187},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['digestive', 'head', 'joint']},
 'indexing': {'chunk_id': 'chunk_000676',
              'chunk_char_len': 146,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
