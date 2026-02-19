from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 보험기간 중이나 보험기간 만료 후 1년 이내에는 보험료 계산에 필요한 경우에 계약자의 서류를 열람할 수 있습니다.\n'
 '제6조(준용규정)\n'
 '이 추가특별약관에 정하지 않은 사항은 보통약관 및 해당특별약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 31},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000152',
              'chunk_char_len': 119,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
