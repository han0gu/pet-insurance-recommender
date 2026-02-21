from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3. 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때\n'
 '- \uf000 제1항에 따라 위험이 증가하거나 감소되는 경우 납입보험료가 변경될 수 있으며,\n'
 '계약내용 변경시점 이후 잔여 보험기간의 보장을 위한 재원인 계약자적립액 등의\n'
 '차이로 계약자가 추가로 납입하여야 할 (또는 반환받을) 금액이 발생할 수 있습'),
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
 'indexing': {'chunk_id': 'chunk_000699',
              'chunk_char_len': 166,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
