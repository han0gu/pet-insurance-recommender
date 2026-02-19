from langchain_core.documents import Document

chunk = Document(
    page_content=('. ③ 제1항에서 정한 계약의 부활이 이루어진 경우라도 계약자 또는 피보험자가 최초 계약 청약시(2회 이상 부활이 이루어진 경우 종전 '
 '모든 부활 청약 포함) 제15조(계약 전 알 릴 의무)를 위반한 경우에는 제17조(알릴 의무 위반의 효과)가 적용됩니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 17},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000107',
              'chunk_char_len': 142,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
