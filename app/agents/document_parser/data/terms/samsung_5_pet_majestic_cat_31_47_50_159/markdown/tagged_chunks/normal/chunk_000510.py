from langchain_core.documents import Document

chunk = Document(
    page_content=('에 따라 보장을 합니다. 또한, 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙\n'
 '한 경우에는 제1회 보험료를 받은 때부터 보장이 개시됩니다. 자동이체 또는 신용카\n'
 '드로 납입하는 경우에는 자동이체신청 또는 신용카드매출승인에 필요한 정보를 제공\n'
 '한 때를 제1회 보험료를 받은 때로 하며, 계약자의 책임 있는 사유로 자동이체 또는\n'
 '매출승인이 불가능한 경우에는 보험료가 납입되지 않은 것으로 봅니다.\n'
 '② 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙하기 전에 보험금 지급사유가'),
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
 'indexing': {'chunk_id': 'chunk_000510',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
