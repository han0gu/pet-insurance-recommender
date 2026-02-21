from langchain_core.documents import Document

chunk = Document(
    page_content=('일자 및 시간 필수) 등)\n'
 '3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관 발행 신분증, 본인이\n'
 '아닌 경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확\n'
 '보된 전자적 수단을 활용한 피보험자 의사표시의 확인방법 포함)4. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류② 제1항 '
 '제2호의 사고증명서는 수의사법 제12조(진단서 등)에서 규정한 내용에 따라 국'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000029',
              'chunk_char_len': 223,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
