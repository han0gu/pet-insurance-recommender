from langchain_core.documents import Document

chunk = Document(
    page_content=('5. 기간과 날짜 관련 용어\n'
 '가. 보험기간: 계약에 따라 보장을 받는 기간을 말합니다. 나. 영업일: 회사가 영업점에서 정상적으로 영업하는 날을 말하며, 토요일, '
 '‘관공서의 공휴일에 관한 규정’에 따른 공휴일(대체공휴일 포함)과 ‘노동절 제정에 관한 법률’ 에 따른 노동절을 제외합니다.\n'
 '제3조(피보험자의 범위)\n'
 '① 이 계약에서 피보험자라 함은 아래에 정한 보험증권에 기재된 피보험자 및 그 가족을 말합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 2},
 'term_type': 'basic',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000010',
              'chunk_char_len': 228,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
