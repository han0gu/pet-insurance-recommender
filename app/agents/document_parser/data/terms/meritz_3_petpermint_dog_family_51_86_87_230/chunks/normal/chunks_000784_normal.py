from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 지급률의 결정\n'
 '1) 한 다리의 3대 관절중 관절 하나에 기능장해가 생기고 다른 관절 하나에 기능장해가 발생한 경우 지급률은 각각 적용하여 합산한다. '
 '2) 1하지(다리와 발가락)의 장해 지급률은 원칙적으로 각 각 합산하되, 지급률은 60% 한도로 한다.\n'
 '10. 손가락의 장해\n'
 '가. 장해의 분류\n'
 '장해의 분류 | 지급률\n'
 '1) 한손의 5개 손가락을 모두 잃었을 때 | 55\n'
 '2) 한손의 첫째 손가락을 잃었을 때 | 15\n'
 '3) 한손의 첫째 손가락 이외의 손가락을 잃었을 때 (손가락 하나마다) | 10'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 220},
 'term_type': 'special',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000784',
              'chunk_char_len': 278,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
