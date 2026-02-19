from langchain_core.documents import Document

chunk = Document(
    page_content=('2) 흉복부장기 또는 비뇨생식기 기능을 잃었을 때 | 75\n'
 '3) 흉복부장기 또는 비뇨생식기 기능에 심한 장해를 남긴 때 | 50\n'
 '4) 흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를 남긴 때 | 30\n'
 '5) 흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 때 | 15\n'
 '나. 장해의 판정기준\n'
 '1) "심장 기능을 잃었을 때" 라 함은 심장 이식을 한 경우를 말한다. 2) "흉복부장기 또는 비뇨생식기 기능을 잃었을 때" 라 함은 '
 '아래의 경우 중 하나 에 해당하는 때를 말한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 146},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['digestive', 'urinary']},
 'indexing': {'chunk_id': 'chunk_000963',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
