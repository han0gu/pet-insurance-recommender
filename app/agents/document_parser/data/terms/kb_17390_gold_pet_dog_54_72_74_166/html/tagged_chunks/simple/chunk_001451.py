from langchain_core.documents import Document

chunk = Document(
    page_content=('기간 종료<br>일 이내에서 계약의 부담보 기간을 적용하고, 유사계약에서 정한 질병과 동일하거<br>나 축소된 범위로 계약의 부담보 설정 '
 '범위를 정합니다'),
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
 'indexing': {'chunk_id': 'chunk_001451',
              'chunk_char_len': 85,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
