from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 척추체(척추뼈 몸통)에 골절 또는 탈구로 4개 이 상의 척추체(척추뼈 몸통)를 유합(아물어 붙음) 또는 고정한 상태 나) '
 '머리뼈(두개골), 제1경추, 제2경추를 모두 유합'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 211},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['joint', 'head']},
 'indexing': {'chunk_id': 'chunk_000748',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.92}},
)
