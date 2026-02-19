from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 지급률의 결정\n'
 '1) 한 다리의 3대 관절 중 관절 하나에 기능장해가 생기고 다른 관절 하나에 기능 장해가 발생한 경우 지급률은 각각 적용하여 합산한다. '
 '2) 1하지(다리와 발가락)의 후유장해 지급률은 원칙적으로 각각 합산하되, 지급률 은 60% 한도로 한다.\n'
 '10. 손가락의 장해\n'
 '가. 장해의 분류\n'
 '장 해 의 분 류 | 지급률(%)\n'
 '1) 한 손의 5개 손가락을 모두 잃었을 때 | 55\n'
 '2) 한 손의 첫째 손가락을 잃었을 때 | 15\n'
 '3) 한 손의 첫째 손가락 이외의 손가락을 잃었을 때(손가락 하나마다) | 10'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 145},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage',
            'risk_domains': ['joint', 'digestive', 'other']},
 'indexing': {'chunk_id': 'chunk_000948',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
