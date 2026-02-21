from langchain_core.documents import Document

chunk = Document(
    page_content=('정하는 후유장해상태가 발생하였을 경우에는 직전까지의 후유장해에 대한 후유\n'
 '장해보험금이 지급된 것으로 보고 최종 후유장해 상태에 해당되는 후유장해보험| 금에서 이를 차감하여 | 지급합니다. |\n'
 '| --- | --- |'),
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
 'indexing': {'chunk_id': 'chunk_000227',
              'chunk_char_len': 120,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
