from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사가 보상한 금액이 피보험자가 입은<br>손해의 일부인 경우에는 피보험자의 권리를 침해하지 않는 범위내에서 그 권리를 '
 '가집<br>니다.<br>1. 피보험자가 제3자로부터 손해배상을 받을 수 있는 경우에는 그 손해배상청구권<br>2. 피보험자가 손해배상을 '
 '함으로써 대위 취득하는 것이 있을 경우에는 그 대위권<br>② 계약자 또는 피보험자는 제1항에 의하여 회사가 취득한 권리를 행사하거나 '
 '지키는 것<br>에 관하여 조치를 하여야 하며, 또한 회사가 요구하는 증거 및 서류를 제출하여야 합니<br>다'),
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
 'indexing': {'chunk_id': 'chunk_000255',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
