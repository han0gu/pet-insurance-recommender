from langchain_core.documents import Document

chunk = Document(
    page_content=('② 이 특별약관이 의무보험이 아니고 다른 의무보험이 있는 경우에는 다른 의무보험에서 보상되는 금액(피보험자가 가입을 하지 않은 경우에는 '
 '보상될 것으로 추정되는 금액)을'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 25},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 93,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
