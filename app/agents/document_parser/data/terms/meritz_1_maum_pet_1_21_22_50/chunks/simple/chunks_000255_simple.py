from langchain_core.documents import Document

chunk = Document(
    page_content=('제5조(준용규정)\n'
 '① 이 특별약관에서 정하지 않은 사항에 대하여는 전환대상계약의 약관, 소득세법 등 관련 법규에서 정하는 바에 따릅니다. ② 소득세법 등 '
 '관련법규가 제·개정 또는 폐지되는 경우 변경된 법령을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 47},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000255',
              'chunk_char_len': 121,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
