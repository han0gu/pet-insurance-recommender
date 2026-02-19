from langchain_core.documents import Document

chunk = Document(
    page_content=('제10조(지급보험금의 계산)\n'
 '① 동일한 반려동물과 동일한 사고에 관하여 보험금을 지급하는 다른 계약(공제계약을 포 함합니다)이 있을 경우 각 계약에 대하여 다른 '
 '계약이 없는 것으로 하여 각각 산출한 지급보험금의 합계액이 피보험자가 부담한 비용금액을 초과할 때에는 아래에 따라 보 험금을 '
 '지급합니다.\n'
 '피보험자가 부담한 총 비용금액\n'
 '×\n'
 '이 계약의 지급보험금\n'
 '다른 계약이 없는 것으로 하여 각각\n'
 '계산한 지급보험금의 합계액'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 8},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000046',
              'chunk_char_len': 232,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
