from langchain_core.documents import Document

chunk = Document(
    page_content=('제20조(계약자의 임의해지)\n'
 '계약자는 손해가 발생하기 전에는 언제든지 계약을 해지할 수 있습니다. 다만, 타인을 위 한 계약의 경우에는 계약자는 그 타인의 동의를 '
 '얻거나 보험증권을 소지한 경우에 한하여 계약을 해지할 수 있습니다.\n'
 '제21조(준용규정)\n'
 '이 특별약관에서 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 30},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000185',
              'chunk_char_len': 171,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
