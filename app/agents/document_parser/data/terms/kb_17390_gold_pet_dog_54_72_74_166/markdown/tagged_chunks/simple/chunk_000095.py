from langchain_core.documents import Document

chunk = Document(
    page_content=('외되는 질병으로 추가 진단(단순 건강검진 제외) 또는 치료 사실이 없을 경우, 청\n'
 '약일로부터 5년이 지난 이후에는 이 약관에 따라 보장합니다.\n'
 '\uf000 제5항의 추가 진단(단순 건강검진 제외) 또는 치료 사실이 없는 경우는 다음 각 호- \n'
 '- 의 경우를 포함합니다.\n'
 '- 1. 검진결과 추가검사 또는 치료가 필요하지 않았던 경우'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000095',
              'chunk_char_len': 179,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
