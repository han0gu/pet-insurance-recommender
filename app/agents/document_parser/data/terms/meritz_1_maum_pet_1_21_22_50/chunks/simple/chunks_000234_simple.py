from langchain_core.documents import Document

chunk = Document(
    page_content=('제6조(보험금의 청구)\n'
 '지정대리청구인은 회사가 정하는 방법에 따라 다음의 서류를 제출하고 보험금을 청구하여 야 합니다.\n'
 '1. 청구서(회사양식) 2. 사고증명서 3. 신분증(주민등록증이나 운전면허증 등 사진이 붙은 정부기관발행 신분증) 4. 피보험자 및 '
 '지정대리청구인의 가족관계등록부(가족관계증명서) 및 주민등록등본 5. 기타 지정대리청구인이 보험금 등의 수령에 필요하여 제출하는 서류\n'
 '제7조(준용규정)\n'
 '이 특약에서 정하지 않은 사항에 대하여는 보통약관 및 해당 특별약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 43},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000234',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
