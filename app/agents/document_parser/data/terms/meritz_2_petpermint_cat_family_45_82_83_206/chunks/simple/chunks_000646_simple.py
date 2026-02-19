from langchain_core.documents import Document

chunk = Document(
    page_content=('4. 씹어먹거나 말하는 장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 씹어먹는 기능과 말하는 기능 모두에 심한 장 해를 남긴 때 | 100\n'
 '2) 씹어먹는 기능에 심한 장해를 남긴 때 | 80\n'
 '3) 말하는 기능에 심한 장해를 남긴 때 | 60\n'
 '4) 씹어먹는 기능과 말하는 기능 모두에 뚜렷한 장해를 남긴 때 | 40\n'
 '5) 씹어먹는 기능 또는 말하는 기능에 뚜렷한 장 해를 남긴 때 | 20\n'
 '6) 씹어먹는 기능과 말하는 기능 모두에 약간의 장해를 남긴 때 | 10'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 181},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['dental', 'head']},
 'indexing': {'chunk_id': 'chunk_000646',
              'chunk_char_len': 260,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
