from langchain_core.documents import Document

chunk = Document(
    page_content=('직접적인 목적으로 동물병원에 통원 또는 입원하여 수의사에게 치료를 받은<br>때에는 피보험자가 부담한 반려동물의 치료비를 보통약관 '
 '제4조(보험금의 지급사유)에<br>따라 피보험자에게 치료비보험금으로 보상하여 드립니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000289',
              'chunk_char_len': 120,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
