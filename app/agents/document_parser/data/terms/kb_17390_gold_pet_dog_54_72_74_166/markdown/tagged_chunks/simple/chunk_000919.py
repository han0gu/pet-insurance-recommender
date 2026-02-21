from langchain_core.documents import Document

chunk = Document(
    page_content=('- 이 되는 때에는 그 내고정물 등이 제거된 후에 장해를 평가한다. 단, 제\n'
 '- 거가 불가능한 경우에는 고정물 등이 있는 상태에서 장해를 평가한다.\n'
 '- 2) 관절을 사용하지 않아 발생한 일시적인 기능장해(예를 들면 캐스트로 환\n'
 '- 부를 고정시켰기 때문에 치유 후의 관절에 기능장해가 발생한 경우)는\n'
 '- 장해로 평가하지 않는다.\n'
 '- 3) ‘발가락을 잃었을 때’라 함은 첫째 발가락에서는 지관절부터 심장에 가\n'
 '- 까운 쪽을, 나머지 네 발가락에서는 제1지관절(근위지관절)부터(제1지'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000919',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
