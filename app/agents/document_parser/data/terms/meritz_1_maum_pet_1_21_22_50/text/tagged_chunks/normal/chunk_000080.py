from langchain_core.documents import Document

chunk = Document(
    page_content=('수 없는 경우에는 이 계약은 그 때부터 효력이 없습니다.제5관 보험료의 납입제25조(제1회 보험료 및 회사의 보장개시)① 회사는 계약의 '
 '청약을 승낙하고 제1회 보험료를 받은 때부터 이 약관이 정한 바에 따\n'
 '라 보장을 합니다. 또한, 회사가 청약과 함께 제1회 보험료를 받은 후 승낙한 경우에도\n'
 '제1회 보험료를 받은 때부터 보장이 개시됩니다. 자동이체 또는 신용카드로 납입하는\n'
 '경우에는 자동이체신청 또는 신용카드매출승인에 필요한 정보를 제공한 때를 제1회 보'),
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
 'indexing': {'chunk_id': 'chunk_000080',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
