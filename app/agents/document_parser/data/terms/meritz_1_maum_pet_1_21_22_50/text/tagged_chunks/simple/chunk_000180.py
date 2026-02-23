from langchain_core.documents import Document

chunk = Document(
    page_content=('② 회사는 특별약관 제4조(보험의 목적의 증가 감소 또는 교체) 제3항에도 불구하고 보험\n'
 '료가 정산되기 이전일지라도 새로이 증가 또는 교체된 보험의 목적에 대해 생긴 손해\n'
 '를 보상합니다.제2조(보험의 목적의 명부)계약자는 항상 보험의 목적의 명부를 비치하여 회사가 열람을 요구할 경우에는 이에 따라\n'
 '야 합니다.제3조(예치보험료)예치보험료는 계약체결일 이전 1개월 동안 1일 평균 보험의 목적의 수에 정해진 보험요율'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000180',
              'chunk_char_len': 231,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
