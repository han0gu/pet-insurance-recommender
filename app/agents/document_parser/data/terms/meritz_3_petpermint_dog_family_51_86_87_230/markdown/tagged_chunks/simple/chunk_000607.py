from langchain_core.documents import Document

chunk = Document(
    page_content=('- 경우를 말하며, 후각감퇴는 장해의 대상으로 하지 않는다.\n'
 '- 3) 양쪽 코의 후각기능은 후각인지검사, 후각역치검사\n'
 '- 등을 통해 6개월 이상 고정된 후각의 완전손실이 확\n'
 '- 인되어야 한다.\n'
 '- 4) 코의 추상(추한 모습)장해를 수반한 때에는 기능장해의\n'
 '- 지급률과 추상(추한 모습)장해의 지급률을 합산한다.\n'
 '# 4. 씹어먹거나 말하는 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 씹어먹는 기능과 말하는 기능 모두에 심한 장 해를 남긴 때 | 100 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000607',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
