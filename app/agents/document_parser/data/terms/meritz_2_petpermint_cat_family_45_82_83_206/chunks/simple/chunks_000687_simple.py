from langchain_core.documents import Document

chunk = Document(
    page_content=('장해의 분류 | 지급률\n'
 '를 남긴 때 6) 한팔의 3대관절중 관절 하나의 기능에 약간의 장해 를 남긴 때 7) 한팔에 가관절이 남아 뚜렷한 장해를 남긴 때 8) '
 '한팔에 가관절이 남아 약간의 장해를 남긴 때 9) 한팔의 뼈에 기형을 남긴 때 | 5 20 10 5\n'
 '나. 장해판정기준'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 191},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000687',
              'chunk_char_len': 154,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.92}},
)
