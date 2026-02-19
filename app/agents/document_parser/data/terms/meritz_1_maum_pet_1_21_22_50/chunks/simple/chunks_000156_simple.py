from langchain_core.documents import Document

chunk = Document(
    page_content=('제10조(보험금의 분담)\n'
 '① 회사는 이 특별약관에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합니다)이 있을 경우 각 계약에 대하여 다른 '
 '계약이 없는 것으로 하여 각각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 아래에 따라 손해를 보상합니다. 이 특 별약관과 다른 '
 '계약이 모두 의무보험인 경우에도 같습니다.\n'
 '손해액 ×\n'
 '이 계약의 보상책임액\n'
 '다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 합계액\n'
 '【사례】'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 25},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000156',
              'chunk_char_len': 246,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
