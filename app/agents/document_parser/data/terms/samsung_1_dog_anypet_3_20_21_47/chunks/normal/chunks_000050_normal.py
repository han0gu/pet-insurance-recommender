from langchain_core.documents import Document

chunk = Document(
    page_content=('※ 이 경우 청약철회는 2025년 1월 20일까지 가능합니다. 2. 청약일 : 2025년 1월 3일 / 보험증권을 받은 날 : 2025년 '
 '1년 20일인 경우\n'
 '- 보험증권을 받은 날부터 15일 : 2025년 2월 4일\n'
 '- 청약을 한 날로부터 30일 : 2025년 2월 2일 (←먼저도래)\n'
 '※ 이 경우 청약철회는 2025년 2월 2일까지 가능합니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 11},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000050',
              'chunk_char_len': 193,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
