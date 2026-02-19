from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자 또는 피보험자가 알리지 않은 경우 회사가 알고 있는 최종 의 주소 또는 연락처로 등기우편 등 우편물에 대한 기록이 남는 '
 '방법으로 회사가 알린 사항은 일반적으로 도달에 필요한 기간이 지난 때에는 계약자 또는 피보험자에게 도달 한 것으로 봅니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 28},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000173',
              'chunk_char_len': 145,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
