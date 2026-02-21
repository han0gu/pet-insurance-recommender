from langchain_core.documents import Document

chunk = Document(
    page_content=('- 후유장해보험금이 지급된 것으로 보고 최종 후유장해 상태에 해당되는 후유장해\n'
 '# 보험금에서 이를 차감하여 지급합니다.| 예 시 | 장해지급률 계산 |\n'
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
 'indexing': {'chunk_id': 'chunk_000257',
              'chunk_char_len': 98,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
