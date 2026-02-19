from langchain_core.documents import Document

chunk = Document(
    page_content=('12. 흉ㆍ복부장기 및 비뇨생식기의 장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 심장 기능을 잃었을 때 | 100\n'
 '2) 흉복부장기 또는 비뇨생식기 기능을 잃었을 때 | 75\n'
 '3) 흉복부장기 또는 비뇨생식기 기능에 심한 장 해를 남긴 때 | 50\n'
 '4) 흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를 남긴 때 | 30\n'
 '5) 흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 때 | 15\n'
 '나. 장해의 판정기준'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 199},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000723',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
