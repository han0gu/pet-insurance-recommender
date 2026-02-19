from langchain_core.documents import Document

chunk = Document(
    page_content=('. 3) “팔”이라 함은 어깨관절(견관절)부터 손목관절(완 관절)까지를 말한다. 4) “팔의 3대관절”이라 함은 어깨관절(견관절), '
 '팔꿈치 관절(주관절), 손목관절(완관절)을 말한다. 5) “한팔의 손목이상을 잃었을 때”라 함은 손목관절 (완관절)부터(손목관절 포함) '
 '심장에 가까운 쪽에서 절단된 때를 말하며, 팔꿈치관절(주관절) 상부에서 절단된 경우도 포함한다. 6) 팔의 관절기능 장해 평가는 팔의 '
 '3대관절의 관절운동 범위 제한 등으로 평가한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 216},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000765',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
