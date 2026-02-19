from langchain_core.documents import Document

chunk = Document(
    page_content=('. 마. 보험금 분담: 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제 계약을 포함합니다)이 있을 경우 비율에 따라 '
 '손해를 보상합니다. 바. 대위권: 회사가 보험금을 지급하고 취득하는 법률상의 권리를 말합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 22},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000138',
              'chunk_char_len': 127,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
