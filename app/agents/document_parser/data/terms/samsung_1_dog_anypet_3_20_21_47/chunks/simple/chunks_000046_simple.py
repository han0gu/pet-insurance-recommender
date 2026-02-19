from langchain_core.documents import Document

chunk = Document(
    page_content='. ④ 이미 성립한 계약을 연장하거나 변경하는 경우에는 회사는 보험증권에 그 사실을 기재함으로써 보 험증권의 교부에 대신할 수 있습니다.',
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 10},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000046',
              'chunk_char_len': 76,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
