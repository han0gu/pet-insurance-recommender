from langchain_core.documents import Document

chunk = Document(
    page_content=('![image](/image/placeholder)\n'
 '< 발가락 >22312. 흉ㆍ복부장기 및 비뇨생식기의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 1) 심장 기능을 잃었을 때 | 100 |\n'
 '| 2) 흉복부장기 또는 비뇨생식기 기능을 잃었을 때 | 75 |\n'
 '| 3) 흉복부장기 또는 비뇨생식기 기능에 심한 장 해를 남긴 때 | 50 |\n'
 '| 4) 흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를 남긴 때 | 30 |\n'
 '| 5) 흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 때 | 15 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000674',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
