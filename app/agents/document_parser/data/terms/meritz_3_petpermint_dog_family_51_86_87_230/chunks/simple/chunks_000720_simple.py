from langchain_core.documents import Document

chunk = Document(
    page_content=('1) "코의 호흡기능을 완전히 잃었을 때"라 함은 일상생 활에서 구강호흡의 보조를 받지 않는 상태에서 코로 숨쉬는 것만으로 정상적인 '
 '호흡을 할 수 없다는 것이 비강통기도검사 등 의학적으로 인정된 검사로 확인되 는 경우를 말한다. 2) “코의 후각기능을 완전히 잃었을 '
 '때”라 함은 후각 신경의 손상으로 양쪽 코의 후각기능을 완전히 잃은 경우를 말하며, 후각감퇴는 장해의 대상으로 하지 않는다. 3) 양쪽 '
 '코의 후각기능은 후각인지검사, 후각역치검사 등을 통해 6개월 이상 고정된 후각의 완전손실이 확 인되어야 한다'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 206},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head', 'other']},
 'indexing': {'chunk_id': 'chunk_000720',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
