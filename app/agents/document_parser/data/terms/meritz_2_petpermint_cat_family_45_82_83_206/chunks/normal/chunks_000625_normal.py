from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 장해분류별 판정기준\n'
 '1. 눈의 장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 두눈이 멀었을 때 | 100\n'
 '2) 한눈이 멀었을 때 | 50\n'
 '3) 한눈의 교정시력이 0.02 이하로 된 때 | 35\n'
 '4) 한 눈의 교정시력이 0.06 이하로 된 때 | 25\n'
 '5) 한 눈의 교정시력이 0.1 이하로 된 때 | 15\n'
 '6) 한 눈의 교정시력이 0.2 이하로 된 때 | 5\n'
 '7) 한눈의 안구(눈동자)에 뚜렷한 운동장해나 뚜렷한 조절기능장해를 남긴 때 | 10\n'
 '8) 한 눈에 뚜렷한 시야장해를 남긴 때 | 5'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 177},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000625',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
