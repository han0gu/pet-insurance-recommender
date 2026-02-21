from langchain_core.documents import Document

chunk = Document(
    page_content=('립한 후, 금융기관이 영업정지나 파산 등으로 예금을 지급할 수 없게 되면 금융기관을\n'
 '대신하여 예금을 지급하는 제도를 말합니다. 이 보험계약은 예금자보호법에 따라 해약\n'
 '환급금(또는 만기 시 보험금)에 기타지급금을 합한 금액이 1인당 “1억원까지”(본 보험\n'
 '회사의 여타 보호상품과 합산) 보호됩니다. 이와 별도로 본 보험회사 보호상품의 사고\n'
 '보험금을 합산한 금액이 1인당 “1억원까지”보호됩니다. 다만, 보험계약자 및 보험료납'),
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
 'indexing': {'chunk_id': 'chunk_000115',
              'chunk_char_len': 236,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
