from langchain_core.documents import Document

chunk = Document(
    page_content=('제6관 계약의 해지 및 보험료의 환급 등\n'
 '제30조(계약의 해지)\n'
 '계약자는 계약이 소멸하기 전에는 언제든지 계약을 해지할 수 있으며, 이 경우 회사가 환 급하여야 할 보험료가 있을 경우에는 '
 '제33조(보험료의 환급)에 따른 보험료를 계약자에게 지급합니다.\n'
 '제30조의2(위법계약의 해지)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 18},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000113',
              'chunk_char_len': 157,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
