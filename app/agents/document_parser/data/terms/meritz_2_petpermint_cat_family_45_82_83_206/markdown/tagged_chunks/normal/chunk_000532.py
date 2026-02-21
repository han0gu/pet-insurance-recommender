from langchain_core.documents import Document

chunk = Document(
    page_content=('| 2) 코의 후각기능을 완전히 잃었을 때 | 5 |\n'
 '# 나. 장해판정기준- 1) "코의 호흡기능을 완전히 잃었을 때"라 함은 일상생\n'
 '- 활에서 구강호흡의 보조를 받지 않는 상태에서 코로\n'
 '- 숨쉬는 것만으로 정상적인 호흡을 할 수 없다는 것이\n'
 '- 비강통기도검사 등 의학적으로 인정된 검사로 확인되\n'
 '- 는 경우를 말한다.\n'
 '- 2) “코의 후각기능을 완전히 잃었을 때”라 함은 후각\n'
 '- 신경의 손상으로 양쪽 코의 후각기능을 완전히 잃은\n'
 '- 경우를 말하며, 후각감퇴는 장해의 대상으로 하지 않는다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['dental', 'digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000532',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
