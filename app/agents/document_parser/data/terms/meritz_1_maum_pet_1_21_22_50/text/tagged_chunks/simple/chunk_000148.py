from langchain_core.documents import Document

chunk = Document(
    page_content=('해지할 수 있습니다.\n'
 '③ 계약자 또는 피보험자는 주소 또는 연락처가 변경된 경우에는 지체없이 이를 회사에 알\n'
 '려야 합니다. 다만, 계약자 또는 피보험자가 알리지 않은 경우 회사가 알고 있는 최종\n'
 '의 주소 또는 연락처로 등기우편 등 우편물에 대한 기록이 남는 방법으로 회사가 알린\n'
 '사항은 일반적으로 도달에 필요한 기간이 지난 때에는 계약자 또는 피보험자에게 도달\n'
 '한 것으로 봅니다.【계약 후 알릴 의무】상법 제652조에서 정하고 있는 의무. 보험기간 중에 보험계약자 또는 피보험자가'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000148',
              'chunk_char_len': 268,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
