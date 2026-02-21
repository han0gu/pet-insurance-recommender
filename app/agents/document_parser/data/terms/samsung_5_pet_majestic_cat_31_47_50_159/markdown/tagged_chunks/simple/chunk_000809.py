from langchain_core.documents import Document

chunk = Document(
    page_content=('| 5) 한 손의 첫째 손가락의 손가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 10 |\n'
 '| 6) 한 손의 첫째 손가락 이외의 손가락의 손가락뼈 일부를 잃었을 때 또는 뚜렷 한 장해를 남긴 때(손가락 하나마다) | 5 |\n'
 '# 나. 장해판정기준- 1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것이 기능장해의 원인이 되\n'
 '- 는 때에는 그 내고정물 등이 제거된 후에 장해를 평가한다. 단, 제거가 불가능\n'
 '- 한 경우에는 고정물 등이 있는 상태에서 장해를 평가한다.'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000809',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
