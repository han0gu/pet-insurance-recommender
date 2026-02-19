from langchain_core.documents import Document

chunk = Document(
    page_content=('제2조(계약 후의 알릴의무)\n'
 '계약자는 지정계좌의 번호가 변경 또는 거래 정지된 경우에는 이 사실을 즉시 회사에 알려 야 합니다.\n'
 '제3조(준용규정)\n'
 '이 추가특별약관에 정하지 않은 사항은 보통약관 및 보험료자동납입 특별약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 36},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000202',
              'chunk_char_len': 129,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
