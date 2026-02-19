from langchain_core.documents import Document

chunk = Document(
    page_content=('. ④ 회사는 제1항에 따른 권리가 계약자 또는 피보험자와 생계를 같이 하는 가족에 대한 것인 경우에는 그 권리를 취득하지 못합니다. '
 '다만, 손해가 그 가족의 고의로 인하여 발생한 경우에는 그 권리를 취득합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 27},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000169',
              'chunk_char_len': 118,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
