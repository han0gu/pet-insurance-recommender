from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 제1항에 열거하는 서류 이외의 서류 제출을 요구할 수 있습니다.\n'
 '제4조(준용규정)\n'
 '이 특별약관에서 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 23},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000120',
              'chunk_char_len': 83,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
