from langchain_core.documents import Document

chunk = Document(
    page_content=('경우에는 자동이체신청 또는 신용카드매출승인에 필요한 정보를 제공한 때를 제1회 보\n'
 '험료를 받은 때로 하며, 계약자의 책임 있는 사유로 자동이체 또는 매출승인이 불가능\n'
 '한 경우에는 보험료가 납입되지 않은 것으로 봅니다.\n'
 '② 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙하기 전에 보험금 지급사유가 발\n'
 '생하였을 때에도 보장개시일부터 이 약관이 정하는 바에 따라 보장을 합니다.【보장개시일】회사가 보장을 개시하는 날로서 계약이 성립되고 '
 '제1회 보험료를 받은 날을 말하나,'),
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
 'indexing': {'chunk_id': 'chunk_000081',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
