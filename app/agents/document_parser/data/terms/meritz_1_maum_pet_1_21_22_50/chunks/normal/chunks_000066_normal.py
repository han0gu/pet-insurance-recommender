from langchain_core.documents import Document

chunk = Document(
    page_content=('③ 제1항에 따라 계약의 해지가 보험금 지급사유 발생 전에 이루어진 경우, 이로 인하여 회사가 환급하여야 할 보험료가 있을 때에는 '
 '제33조(보험료의 환급)에 따른 보험료를 계약자에게 지급합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 11},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000066',
              'chunk_char_len': 107,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
