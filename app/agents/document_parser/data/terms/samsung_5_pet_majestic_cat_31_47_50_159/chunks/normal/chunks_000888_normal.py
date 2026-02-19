from langchain_core.documents import Document

chunk = Document(
    page_content=('1) "코의 호흡기능을 완전히 잃었을 때" 라 함은 일상생활에서 구강호흡의 보조 를 받지 않는 상태에서 코로 숨쉬는 것만으로 정상적인 '
 '호흡을 할 수 없다는 것이 비강통기도검사 등 의학적으로 인정된 검사로 확인되는 경우를 말한다. 2) "코의 후각기능을 완전히 잃었을 때" '
 '라 함은 후각신경의 손상으로 양쪽 코의 후각기능을 완전히 잃은 경우를 말하며, 후각감퇴는 장해의 대상으로 하지 않 는다. 3) 양쪽 코의 '
 '후각기능은 후각인지검사, 후각역치검사 등을 통해 6개월 이상 고정 된 후각의 완전손실이 확인되어야 한다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 138},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000888',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
