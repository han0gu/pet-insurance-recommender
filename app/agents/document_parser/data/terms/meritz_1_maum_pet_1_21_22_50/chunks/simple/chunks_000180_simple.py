from langchain_core.documents import Document

chunk = Document(
    page_content=('. 계약자의 전자문서 수신이 확인되기 전까지는 그 전자문서는 송신되지 않은 것으로 봅니다. 회사는 전자문 서가 수신되지 않은 것을 확인한 '
 '경우에는 서면(등기우편 등)으로 다시 알려드립니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 29},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000180',
              'chunk_char_len': 105,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
