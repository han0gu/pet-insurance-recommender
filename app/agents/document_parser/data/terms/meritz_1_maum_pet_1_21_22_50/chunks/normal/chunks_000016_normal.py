from langchain_core.documents import Document

chunk = Document(
    page_content=('피보험자가 부담한 치료비 | 10만원 | 20만원 | 40만원\n'
 '1일당 예상 치료비보험금(A) | 4만9천원(*) | 11만9천원(**) | 20만원(***)\n'
 '연간 잔여 보상한도(B) | 30만원 | 25만1천원 | 13만2천원\n'
 '최종 치료비보험금 (C=min(A,B)) | 4만9천원 | 11만9천원 | 13만2천원\n'
 '(*) [(10만원 - 3만원)×70%, 20만원] 중 적은금액 (**) [(20만원 - 3만원)×70%, 20만원] 중 적은금액 '
 '(***) [(40만원 - 3만원)×70%, 20만원] 중 적은금액'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 3},
 'term_type': 'basic',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000016',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
