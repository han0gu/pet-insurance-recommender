from langchain_core.documents import Document

chunk = Document(
    page_content=('| 5) 한 발의 5개 발가락 모두의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 20 |\n'
 '| 6) 한 발의 첫째 발가락의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 8 |\n'
 '| 7) 한 발의 첫째 발가락 이외의 발가락의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때(발가락 하나마다) | 3 |\n'
 '# 나.# 장해판정기준- 1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것이 기능장해의 원인\n'
 '- 이 되는 때에는 그 내고정물 등이 제거된 후에 장해를 평가한다. 단, 제'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000918',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
