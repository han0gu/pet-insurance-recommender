from langchain_core.documents import Document

chunk = Document(
    page_content=('| 고지사항이 청약서에 제대로 확인하시기 | 기재되어 있는지 반드시 바랍니다. |\n'
 '| 용 어 풀 이 해지 현재 유지되고 있는 계약 또는 효력이 상실된 계약을 장래를 향하여 소멸시키 보 | 용 어 풀 이 해지 현재 '
 '유지되고 있는 계약 또는 효력이 상실된 계약을 장래를 향하여 소멸시키 보 |\n'
 '거나 계약유지 의사를 포기하여 만기일 이전에 계약관계를 청산하는 것- 제17조(사기에 의한 계약)\n'
 '- \uf000 계약자 또는 피보험자가 대리진단, 약물사용을 수단으로 진단절차를 통과하거나'),
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
 'indexing': {'chunk_id': 'chunk_000090',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
