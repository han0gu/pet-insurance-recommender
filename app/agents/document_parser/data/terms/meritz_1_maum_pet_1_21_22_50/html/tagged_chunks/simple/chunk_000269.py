from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사가 최초계약 체결당시에 그 사실을 알았거나 과실로 알지 못하였을 때<br>2. 회사가 그 사실을 안 날부터 1개월 이상 지났거나 '
 '또는 제1회 보험료를 받은 때부<br>터 보험금 지급사유가 발생하지 않고 2년이 지났을 때<br>3. 최초계약을 체결한 날부터 3년이 '
 '지났을 때<br>4'),
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
 'indexing': {'chunk_id': 'chunk_000269',
              'chunk_char_len': 161,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
