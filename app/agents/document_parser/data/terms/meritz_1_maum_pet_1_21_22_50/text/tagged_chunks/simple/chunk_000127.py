from langchain_core.documents import Document

chunk = Document(
    page_content=('법 제657조 제1항에 의해 보험사고의 발생을 회사에 알린 경우에는 제3조(보상하는 손\n'
 '해) 제3항 제1호 및 제2호 ‘다’목 또는 ‘라’목의 비용에 대하여 보상한도액을 한도로 보\n'
 '상하여 드립니다.제6조(보험금의 청구)피보험자가 보험금을 청구할 때에는 다음의 서류를 회사에 제출하여야 합니다.1. 보험금 '
 '청구서(회사양식)\n'
 '2. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증, 본인이 아닌\n'
 '경우에는 본인의 인감증명서, 본인서명사실확인서 또는 안전성과 신뢰성이 확보된 전자'),
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
 'indexing': {'chunk_id': 'chunk_000127',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
