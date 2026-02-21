from langchain_core.documents import Document

chunk = Document(
    page_content=('관에서 정한 보험금 지급사유가 더이상 발생할 수 없는 경우에는 "보험료 및 해약환\n'
 '급금 산출방법서" 에 정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약\n'
 '자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없# 습니다.# ② 보험의 목적이 다수인 경우 제1항은 '
 '보험의 목적별로 각각 적용합니다.# 제20조 (제1회 보험료 및 회사의 보장개시)① 회사는 특별약관의 청약을 승낙하고 제1회 보험료를 '
 '받은 때부터 이 약관이 정한 바'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000509',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
