from langchain_core.documents import Document

chunk = Document(
    page_content=('. 2. 청약일 : 2025년 1월 3일 / 보험증권을 받은 날 : 2025년 1년 20일인 경우 - 보험증권을 받은 날부터 15일 : '
 '2025년 2월 4일 - 청약을 한 날로부터 30일 : 2025년 2월 2일 (←먼저도래) ※ 이 경우 청약철회는 2025년 2월 '
 '2일까지 가능합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000047',
              'chunk_char_len': 160,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
