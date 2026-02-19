from langchain_core.documents import Document

chunk = Document(
    page_content=('제10조(보험금의 분담)\n'
 '① 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약(공제계약을 포함합니다)이 있을 경 우 각 계약에 대하여 다른 계약이 없는 '
 '것으로 하여 각각 산출한 보상책임액의 합계액이 손해액을 초과할 때에는 회사는 아래에 따라 손해를 보상합니다.\n'
 '다른 계약이 없을 때 이 계약의 보상책임액 손해액(피보험자가 부담한 총비용) × 다른 계약이 없는 것으로 하여 각각 계산한 보상책임액의 '
 '합계액'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 9},
 'term_type': 'basic',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000036',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
