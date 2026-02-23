from langchain_core.documents import Document

chunk = Document(
    page_content=('급금 산출방법서”에 정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약\n'
 '자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없\n'
 '습니다.# 제20조 (제1회 보험료 및 회사의 보장개시)① 회사는 특별약관의 청약을 승낙하고 제1회 보험료를 받은 때부터 이 약관이 정한 '
 '바\n'
 '에 따라 보장을 합니다. 또한, 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙\n'
 '한 경우에는 제1회 보험료를 받은 때부터 보장이 개시됩니다. 자동이체 또는 신용카'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000357',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
