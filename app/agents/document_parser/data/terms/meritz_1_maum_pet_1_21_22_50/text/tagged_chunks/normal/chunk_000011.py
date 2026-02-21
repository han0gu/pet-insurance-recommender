from langchain_core.documents import Document

chunk = Document(
    page_content=('자기부담금을 차감한 후, 보험증권에 기재된 보상비율을 곱한 금액을 보험증권에서 정\n'
 '한 1일당 지급 한도를 적용하여 보상합니다. 다만, 연간 지급하는 총 보험금은 보험증\n'
 '권에 기재된 연간 총 보상한도액을 한도로 합니다.| 항목 | 자기부담금 | 지급 한도 |\n'
 '| --- | --- | --- |\n'
 '| 통원 또는 입원하는 경우 | 1일당 ( )원 | 1일당 ( )원 / 연간 ( )원 |\n'
 '【보험금 지급금액 예시】아래의 경우는 이해를 돕기 위한 예시이며, 자기부담금,'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000011',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
