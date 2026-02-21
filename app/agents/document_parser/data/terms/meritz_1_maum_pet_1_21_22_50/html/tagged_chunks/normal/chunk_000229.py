from langchain_core.documents import Document

chunk = Document(
    page_content=(". 제3조(보상하는 손해) 제1호의 손해배상금 : 보상한도액을 한도로 보상하되, 매회의</p><footer id='49' "
 "style='font-size:14px'>- 24 -</footer><p id='50' data-category='paragraph' "
 "style='font-size:14px'>사고마다 자기부담금 3만원을 초과하는 경우에 한하여 그 초과한 부분만 "
 "보상합니<br>다.</p><br><p id='51' data-category='list' style='font-size:14px'>2"),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'deductible', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000229',
              'chunk_char_len': 279,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
