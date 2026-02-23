from langchain_core.documents import Document

chunk = Document(
    page_content=('험자가 입은 손해의 일부인 경우에는 피보험자의 권리를 침해하지 않는 범위 내 약- 123 -질KB 금쪽같은 '
 '펫보험(강아지)(무배당)(26.01) 123병도성![image](/image/placeholder)\n'
 '상![image](/image/placeholder)\n'
 '해![image](/image/placeholder)\n'
 '에서 그 권리를 취득합니다.- 1. 피보험자가 제3자로부터 손해배상을 받을 수 있는 경우에는 그 손해배상청구권\n'
 '- 2. 피보험자가 손해배상을 함으로써 대위 취득하는 것이 있을 경우에는 그 대위권'),
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
 'indexing': {'chunk_id': 'chunk_000693',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
