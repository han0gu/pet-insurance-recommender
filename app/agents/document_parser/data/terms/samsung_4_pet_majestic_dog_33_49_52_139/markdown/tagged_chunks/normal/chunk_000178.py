from langchain_core.documents import Document

chunk = Document(
    page_content=('지급받지 않고 2024년 4월 1일에 일시에 지급받는 경우# - 지급액 = Max(①, ②)- ① 2천만원 + '
 '2천만원÷(1+평균공시이율) + 2천만원÷(1+평균공시이율)2\n'
 '- ② 2천만원 + 2천만원÷(1+보장부분 적용이율) + 2천만원÷(1+보장부분 적용이율)2\n'
 '# 제12조 (주소변경통지)- ① 계약자(보험수익자가 계약자와 다른 경우 보험수익자를 포함합니다)는 주소 또는 연락\n'
 '- 처가 변경된 경우에는 지체없이 그 변경내용을 회사에 알려야 합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000178',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
