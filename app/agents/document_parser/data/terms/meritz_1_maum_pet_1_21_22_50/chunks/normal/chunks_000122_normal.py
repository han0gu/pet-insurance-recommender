from langchain_core.documents import Document

chunk = Document(
    page_content=('【설명】보험사가 해지권을 행사하는 경우 위의 ‘청구일’은 보험사의 해지 의사표시 (서면, 전자우편, 휴대전화 문자메시지 또는 이에 준하는 '
 '전자적 의사표시 포함)가 보험 계약자 또는 그의 대리인에게 도달한 날로 봅니다.\n'
 '제7관 분쟁의 조정 등\n'
 '제34조(분쟁의 조정)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 19},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000122',
              'chunk_char_len': 147,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
