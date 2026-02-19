from langchain_core.documents import Document

chunk = Document(
    page_content=('<관련법규>\n'
 '[금융소비자보호에 관한 법률 제46조(청약의 철회)에서 정한 청약철회가능 기간] 일반금융소비자가 상법 제640조에 따른 보험증권을 받은 '
 '날부터 15일과 청약을 한 날부터 30일 중 먼저 도래하는 기간을 말합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 41},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000094',
              'chunk_char_len': 125,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
