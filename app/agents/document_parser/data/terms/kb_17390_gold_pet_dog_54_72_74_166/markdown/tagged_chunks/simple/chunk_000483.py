from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 청약서의 기재사항을 변경하고자 할 때 또는 변경이 생겼음을 알았을 때\n'
 '- 2. 이 계약에서 보장하는 위험과 동일한 위험을 보장하는 계약을 다른 보험자와\n'
 '- 체결하고자 할 때 또는 이와 같은 계약이 있음을 알았을 때\n'
 '- 3. 반려동물을 양도할 때\n'
 '4. 위 이외에 위험이 뚜렷이 변경되거나 변경되었음을 알았을 때'),
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
 'indexing': {'chunk_id': 'chunk_000483',
              'chunk_char_len': 177,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
