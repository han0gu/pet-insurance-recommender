from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 보험종목 2. 보험기간 3. 보험료 납입주기, 납입방법 및 납입기간 4. 계약자, 피보험자 5. 보험가입금액, 보험료 등 기타 '
 '계약의 내용'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 13},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000061',
              'chunk_char_len': 80,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
