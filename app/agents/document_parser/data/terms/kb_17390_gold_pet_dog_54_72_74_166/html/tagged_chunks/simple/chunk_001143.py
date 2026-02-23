from langchain_core.documents import Document

chunk = Document(
    page_content=('보험증권에 기재된 반려동물을 말하며, 이 계약에서 가입 가능한 반려동물은 대한민국 내에서 피보험자와 거주를 함 께하고 있는 개(犬)를 '
 '말합니다'),
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
 'indexing': {'chunk_id': 'chunk_001143',
              'chunk_char_len': 79,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
