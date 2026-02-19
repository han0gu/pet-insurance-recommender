from langchain_core.documents import Document

chunk = Document(
    page_content=('② 전환대상계약이 해지(解止) 또는 기타 사유로 효력이 없게 된 경우 또는 전환대상계약 이 제1항에서 정한 조건을 만족하지 않게 된 경우 '
 '이 특별약관은 그 때부터 효력이 없 습니다. ③ 제2조 제1항에 따라 제출된 장애인증명서상 장애예상기간(또는 장애기간)이 종료된 경 '
 '우에는 제3조 제1항에도 불구하고 이 특별약관은 그때부터 효력이 없습니다. ④ 이 특별약관의 계약자는 전환대상계약의 계약자와 동일하여야 '
 '합니다.\n'
 '제2조(제출서류)\n'
 '① 이 특별약관에 가입하고자 하는 계약자는 모든 피보험자 또는 모든 보험수익자의「소득'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 45},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000245',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
