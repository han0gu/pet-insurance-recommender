from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제17조(알릴 의무 위반의 효과)를 준용하여 회사가 보장을 하지 않을 수 있는 경우<br>3. 진단계약에서 보험금 지급사유가 발생할 '
 '때까지 진단을 받지 않은 경우'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000141',
              'chunk_char_len': 92,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
