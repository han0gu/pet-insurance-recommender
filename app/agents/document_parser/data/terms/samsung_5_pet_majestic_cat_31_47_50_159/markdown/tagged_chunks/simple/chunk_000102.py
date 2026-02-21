from langchain_core.documents import Document

chunk = Document(
    page_content=('- 계약자에게 지급하고, 이 계약은 더 이상 효력이 없습니다.\n'
 '<용어풀이>[계약자적립액]\n'
 '장래의 보험금, 해약환급금 등을 지급하기 위하여 계약자가 납입한 보험료 중 일정액을 회사가 적\n'
 '립해 둔 금액을 말합니다.제5관 보험료의 납입- 41 -# 제 27조 (제1회 보험료 및 회사의 보장개시)- ① 회사는 계약의 청약을 '
 '승낙하고 제1회 보험료를 받은 때부터 이 약관이 정한 바에 따\n'
 '- 라 보장을 합니다. 또한, 회사가 청약과 함께 제1회 보험료를 받은 후 승낙한 경우에'),
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
 'indexing': {'chunk_id': 'chunk_000102',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
