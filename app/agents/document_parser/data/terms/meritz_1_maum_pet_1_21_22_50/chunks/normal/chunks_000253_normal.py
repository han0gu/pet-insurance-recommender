from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 전환대상계약에 이 특별약관이 부가된 이후 제4조(전환 취소)에 따라 전환을 취소한 경 우 또는 전환대상계약이 제1조(특별약관의 '
 '적용범위) 제1항 제2호에서 정한 조건을 만 족하지 않아 이 특별약관의 효력이 없어진 경우 해당 전환대상계약에는 이 특별약관을'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 46},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000253',
              'chunk_char_len': 143,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
