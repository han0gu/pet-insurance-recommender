from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 1) 두팔의 손목이상을 잃었을 때 | 100 |\n'
 '| 2) 한팔의 손목이상을 잃었을 때 | 60 |\n'
 '| 3) 한팔의 3대관절중 관절 하나의 기능을 완전히 잃었 을 때 | 30 |\n'
 '| 4) 한팔의 3대관절중 관절 하나의 기능에 심한 장해를 남 긴 때 | 20 |\n'
 '| 5) 한팔의 3대관절중 관절 하나의 기능에 뚜렷한 장해 | 10 |\n'
 '215| 장해의 분류 | 지급률 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000641',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
