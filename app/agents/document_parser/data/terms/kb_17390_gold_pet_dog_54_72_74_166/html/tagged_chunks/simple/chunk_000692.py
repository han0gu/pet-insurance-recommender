from langchain_core.documents import Document

chunk = Document(
    page_content=('제2항에 따라 환경성질환입원일당을 계속 지급합니다.<br>\uf000 피보험자가 정당한 이유없이 입원기간 중 의사의 지시를 따르지 않은 '
 '때에는 회<br>사는 환경성질환입원일당의 전부 또는 일부를 지급하지 않습니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000692',
              'chunk_char_len': 113,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
