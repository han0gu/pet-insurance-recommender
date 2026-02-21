from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하는 경우에는 자동이체신청 또는 신용카드매출승인에 필요한 정보를 제공한 때를 제\n'
 '- 1회 보험료를 받은 때로 하며, 계약자의 책임 있는 사유로 자동이체 또는 매출승인이\n'
 '- 불가능한 경우에는 보험료가 납입되지 않은 것으로 봅니다.\n'
 '- ② 회사가 청약과 함께 제1회 보험료를 받고 청약을 승낙하기 전에 보험금 지급사유가\n'
 '- 발생하였을 때에도 보장개시일부터 이 약관이 정하는 바에 따라 보장을 합니다.\n'
 '# <용어풀이># [보장개시일]회사가 보장을 개시하는 날로서 계약이 성립되고 제1회 보험료를 받은 날을 말하나, 회사가 승낙'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000105',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
