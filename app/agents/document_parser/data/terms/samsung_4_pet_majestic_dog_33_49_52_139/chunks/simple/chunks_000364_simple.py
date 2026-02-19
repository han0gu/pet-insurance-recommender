from langchain_core.documents import Document

chunk = Document(
    page_content=('<관련법규>\n'
 '[의료법 제3조(의료기관)에 규정한 종합병원] 100개 이상의 병상 구비, 병상수에 따라 일정 개수의 진료과목을 갖추고, 각 진료과목마다 '
 '전속하 는 전문의를 둔 병원을 말합니다.\n'
 '제 3조 (특별약관의 소멸)\n'
 '피보험자가 보험기간 중에 사망하였을 경우에는 "보험료 및 해약환급금 산출방법서"에서 정하는 바에 따라 회사가 적립한 사망당시 이 '
 '특별약관의 계약자적립액 및 미경과보험료 를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없습니다.\n'
 '- 72 -\n'
 '1-4. 응급의료 아나필락시스 진단비(연간1회한) 특별약관'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 91,
         'page': 73},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000364',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
