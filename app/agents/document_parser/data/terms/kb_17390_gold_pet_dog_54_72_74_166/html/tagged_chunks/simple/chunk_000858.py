from langchain_core.documents import Document

chunk = Document(
    page_content=('맺을 때에 계약에서 정한 반려동물의 나이에 미달되었거나 초과되었을 경우 병<br>이 특별약관은 무효로 하며 이미 납입한 이 특별약관의 '
 '보험료를 돌려 드립니다'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000858',
              'chunk_char_len': 87,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
