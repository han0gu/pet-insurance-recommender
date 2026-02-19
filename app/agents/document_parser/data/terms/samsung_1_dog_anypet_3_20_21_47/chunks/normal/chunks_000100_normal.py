from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 계약자 또는 피보험자가 임의 해지하는 경우 2. 회사가 제14조(사기에 의한 계약), 제26조(계약의 해지) 또는 '
 '제27조(중대사유로 인한 해지)에 따라 계약을 취소 또는 해지하는 경우 3. 보험료 미납으로 인한 계약의 효력 상실'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 18},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000100',
              'chunk_char_len': 131,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
