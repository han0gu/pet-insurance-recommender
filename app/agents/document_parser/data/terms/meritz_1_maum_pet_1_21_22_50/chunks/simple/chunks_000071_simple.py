from langchain_core.documents import Document

chunk = Document(
    page_content=('제18조(사기에 의한 계약)\n'
 '계약자 또는 피보험자가 사기에 의하여 계약이 성립되었음을 회사가 증명하는 경우에는 계 약일부터 5년 이내(사기사실을 안 날부터 1개월 '
 '이내)에 계약을 취소할 수 있습니다.\n'
 '제 4 관 보험계약의 성립과 유지\n'
 '제19조(보험계약의 성립)'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 12},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000071',
              'chunk_char_len': 145,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
