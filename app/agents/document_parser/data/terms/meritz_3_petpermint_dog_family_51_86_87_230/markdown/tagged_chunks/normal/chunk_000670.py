from langchain_core.documents import Document

chunk = Document(
    page_content=('| 7) 한발의 첫째발가락 이외의 발가락의 발가락 뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남 긴 때(발가락 하나마다) | 3 |\n'
 '# 나. 장해판정기준- 1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것\n'
 '- 이 기능장해의 원인이 되는 때에는 그 내고정물 등이\n'
 '- 제거된 후에 장해를 평가한다. 단, 제거가 불가능한\n'
 '- 경우에는 고정물 등이 있는 상태에서 장해를 평가한\n'
 '- 다.\n'
 '- 2) 관절을 사용하지 않아 발생한 일시적인 기능장해(예를\n'
 '- 들면 캐스트로 환부를 고정시켰기 때문에 치유 후의 관'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000670',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
