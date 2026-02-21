from langchain_core.documents import Document

chunk = Document(
    page_content=('. 영업일: 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토요일, ‘관공서의<br>공휴일에 관한 규정’에 따른 공휴일(대체공휴일 '
 "포함)과 ‘노동절 제정에 관한 법률’<br>에 따른 노동절을 제외합니다.</p><h1 id='19' "
 "style='font-size:14px'>제3조(피보험자의 범위)</h1><br><p id='20' "
 "data-category='paragraph' style='font-size:14px'>① 이 계약에서 피보험자라 함은 아래에 정한 "
 '보험증권에 기재된 피보험자 및 그'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000013',
              'chunk_char_len': 280,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
