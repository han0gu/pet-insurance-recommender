from langchain_core.documents import Document

chunk = Document(
    page_content=('부담금은 각각 보험증권에 기재된 금액을 말합니다.1. 제3조(보상하는 손해) 제1호의 손해배상금 : 보상한도액을 한도로 보상하되, '
 '매회의- 24 -사고마다 자기부담금 3만원을 초과하는 경우에 한하여 그 초과한 부분만 보상합니\n'
 '다.- 2. 제3조(보상하는 손해) 제2호 ‘가’목, ‘나’목 또는 ‘마’목의 비용 : 비용의 전액을 보상\n'
 '- 합니다.\n'
 '- 3. 제3조(보상하는 손해) 제2호 ‘다’목 또는 ‘라’목의 비용 : 이 비용과 제1호에 의한\n'
 '- 보상액의 합계액을 보상한도액내에서 보상합니다.'),
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
 'indexing': {'chunk_id': 'chunk_000135',
              'chunk_char_len': 275,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
