from langchain_core.documents import Document

chunk = Document(
    page_content=('부담한 치료비</td><td>10만원</td><td>20만원</td><td>40만원</td></tr><tr><td>1일당 예상 '
 '치료비보험금(A)</td><td>4만9천원(*)</td><td>11만9천원(**)</td><td>20만원(***)</td></tr><tr><td>연간 '
 '잔여 보상한도(B)</td><td>30만원</td><td>25만1천원</td><td>13만2천원</td></tr><tr><td>최종 '
 '치료비보험금'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000023',
              'chunk_char_len': 235,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
