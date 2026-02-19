from langchain_core.documents import Document

chunk = Document(
    page_content=('제4조(계약 후 알릴 의무)\n'
 '계약자는 거래은행 지정계좌의 번호가 변경 또는 거래 정지된 경우에는 즉시 이 사실을 회 사에 알려야 합니다.\n'
 '제5조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 35},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000198',
              'chunk_char_len': 117,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
