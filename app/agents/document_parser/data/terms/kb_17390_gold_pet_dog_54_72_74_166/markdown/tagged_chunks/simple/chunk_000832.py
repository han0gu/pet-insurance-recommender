from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5) 위 4)에 따라 장해지급률이 결정되었으나 그 이후 보장받을 수 있는 기간(계\n'
 '- 약의 효력이 없어진 경우에는 보험기간이 10년 이상인 계약은 상해 발생일\n'
 '- 또는 질병의 진단확정일부터 2년 이내로 하고, 보험기간이 10년 미만인 계약\n'
 '- 은 상해 발생일 또는 질병의 진단확정일부터 1년 이내)에 장해상태가 더 악\n'
 '- 화된 때에는 그 악화된 장해상태를 기준으로 장해지급률을 결정한다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000832',
              'chunk_char_len': 218,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
