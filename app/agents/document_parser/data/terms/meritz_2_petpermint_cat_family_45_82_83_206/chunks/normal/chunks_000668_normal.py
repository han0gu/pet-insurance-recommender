from langchain_core.documents import Document

chunk = Document(
    page_content=('. 척추체 (척추뼈 몸통)의 압박률은 인접 상ㆍ하부[인접 상ㆍ 하부 척추체(척추뼈 몸통)에 진구성 골절이 있거나, 다발성 척추골절이 있는 '
 '경우에는 골절된 척추와 가 장 인접한 상ㆍ하부] 정상 척추체(척추뼈 몸통)의 전방 높이의 평균에 대한 골절된 척추체(척추뼈 몸 통) 전방 '
 '높이의 감소비를 압박률로 정한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 186},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000668',
              'chunk_char_len': 171,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
