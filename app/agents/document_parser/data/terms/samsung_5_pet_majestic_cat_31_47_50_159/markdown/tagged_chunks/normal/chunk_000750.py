from langchain_core.documents import Document

chunk = Document(
    page_content=('| 2) 코의 후각기능을 완전히 잃었을 때 | 5 |\n'
 '# 나. 장해판정기준- 1) "코의 호흡기능을 완전히 잃었을 때" 라 함은 일상생활에서 구강호흡의 보조\n'
 '- 를 받지 않는 상태에서 코로 숨쉬는 것만으로 정상적인 호흡을 할 수 없다는\n'
 '- 것이 비강통기도검사 등 의학적으로 인정된 검사로 확인되는 경우를 말한다.\n'
 '- 2) "코의 후각기능을 완전히 잃었을 때" 라 함은 후각신경의 손상으로 양쪽 코의\n'
 '- 후각기능을 완전히 잃은 경우를 말하며, 후각감퇴는 장해의 대상으로 하지 않\n'
 '- 는다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['dental', 'digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000750',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
