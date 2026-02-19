from langchain_core.documents import Document

chunk = Document(
    page_content=('<관련법규>\n'
 '[금융소비자보호에 관한 법률 제46조(청약의 철회)에서 정한 청약철회가능 기간] 일반금융소비자가 상법 제640조에 따른 보험증권을 받은 '
 '날부터 15일과 청약을 한 날부터 30일 중 먼저 도래하는 기간을 말합니다.'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 66,
         'page': 35},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000075',
              'chunk_char_len': 125,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
