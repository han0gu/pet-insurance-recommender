from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 제1항의 경우에 회사는 청약서를 접수한 날로부터 30일 이내에 승낙 또는 거절하여야 하며, 승낙한 때에는 금융기관의 해당계좌에서 '
 '제1회 보험료를 받고 보험증권을 드립니 다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 36},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000201',
              'chunk_char_len': 100,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
