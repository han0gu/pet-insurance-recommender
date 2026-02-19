from langchain_core.documents import Document

chunk = Document(
    page_content=('⑨ 제5항에 따라 보험계약이 연장된 경우 계약자는 회사에 재가입 의사를 표시할 수 있 습니다. 회사는 계약자의 재가입 의사가 확인되었을 '
 '때에는 제1항 및 제2항에서 정한 절차에 따라 회사가 재가입 의사를 확인한 날에 판매중인 제2항의 반려동물보험 상품 으로 재가입하는 '
 '것으로 하며, 기존 계약은 해지됩니다'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 106},
 'term_type': 'special',
 'clause': {'clause_type': 'renewal', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000642',
              'chunk_char_len': 170,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
