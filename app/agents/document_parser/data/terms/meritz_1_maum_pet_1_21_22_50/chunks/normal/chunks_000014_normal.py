from langchain_core.documents import Document

chunk = Document(
    page_content=('. ② 회사가 보상하는 비용은 각 항목별 피보험자가 부담한 치료비에서 보험증권에 기재된 자기부담금을 차감한 후, 보험증권에 기재된 '
 '보상비율을 곱한 금액을 보험증권에서 정 한 1일당 지급 한도를 적용하여 보상합니다. 다만, 연간 지급하는 총 보험금은 보험증 권에 기재된 '
 '연간 총 보상한도액을 한도로 합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 3},
 'term_type': 'basic',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000014',
              'chunk_char_len': 170,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
