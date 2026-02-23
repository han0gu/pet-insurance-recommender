from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 연간 지급하는 총 보험금은 보험증<br>권에 기재된 연간 총 보상한도액을 한도로 합니다.</p><br><table '
 "id='27' style='font-size:18px'><thead><tr><td>항목</td><td>자기부담금</td><td>지급 "
 '한도</td></tr></thead><tbody><tr><td>통원 또는 입원하는 경우</td><td>1일당 ( '
 ")원</td><td>1일당 ( )원 / 연간 ( )원</td></tr></tbody></table><br><p id='28'"),
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
 'indexing': {'chunk_id': 'chunk_000020',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
