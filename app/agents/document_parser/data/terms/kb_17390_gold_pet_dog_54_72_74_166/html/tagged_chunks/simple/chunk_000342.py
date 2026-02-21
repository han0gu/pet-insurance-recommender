from langchain_core.documents import Document

chunk = Document(
    page_content=('. 그러나 그 후유장해가 이미 후유장해보험<br>금을 지급받은 동일한 부위에 가중된 때에는 최종 장해상태에 해당하는 '
 '후유장해<br>보험금에서 이미 지급받은 후유장해보험금을 차감하여 지급합니다'),
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
 'indexing': {'chunk_id': 'chunk_000342',
              'chunk_char_len': 106,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
