from langchain_core.documents import Document

chunk = Document(
    page_content=('【관련법규】\n'
 '< 「금융소비자보호에 관한 법률」 제46조(청약의 철회)>에서 정한 일반금융소비자가 청약을 철회할 수 있 는 기간은 아래와 같습니다. '
 '「상법」 제640조에 따른 보험증권을 받은 날부터 15일과 청약을 한 날부터 30일 중 먼저 도래하는 기간\n'
 '【예시】'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 11},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000045',
              'chunk_char_len': 146,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
