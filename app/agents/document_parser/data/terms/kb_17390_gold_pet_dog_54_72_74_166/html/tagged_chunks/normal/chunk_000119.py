from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자 또는 피보험자가 고의 또는 중대한 과실로 제14조(계약 전 알릴 의무)<br>를 위반하고 그 의무가 중요한 사항에 해당하는 '
 '경우<br>2'),
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
 'indexing': {'chunk_id': 'chunk_000119',
              'chunk_char_len': 82,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
