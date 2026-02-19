from langchain_core.documents import Document

chunk = Document(
    page_content=('< 가슴뼈 >\n'
 '< 골반뼈 >\n'
 '8. 팔의 장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 두팔의 손목이상을 잃었을 때 | 100\n'
 '2) 한팔의 손목이상을 잃었을 때 | 60\n'
 '3) 한팔의 3대관절중 관절 하나의 기능을 완전히 잃었 을 때 | 30\n'
 '4) 한팔의 3대관절중 관절 하나의 기능에 심한 장해를 남 긴 때 | 20\n'
 '5) 한팔의 3대관절중 관절 하나의 기능에 뚜렷한 장해 | 10'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 190},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000686',
              'chunk_char_len': 214,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
