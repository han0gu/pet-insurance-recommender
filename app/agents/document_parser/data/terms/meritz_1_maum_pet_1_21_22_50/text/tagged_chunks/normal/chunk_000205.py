from langchain_core.documents import Document

chunk = Document(
    page_content=('습니다.\n'
 '③ 제2조 제1항에 따라 제출된 장애인증명서상 장애예상기간(또는 장애기간)이 종료된 경\n'
 '우에는 제3조 제1항에도 불구하고 이 특별약관은 그때부터 효력이 없습니다.\n'
 '④ 이 특별약관의 계약자는 전환대상계약의 계약자와 동일하여야 합니다.제2조(제출서류)① 이 특별약관에 가입하고자 하는 계약자는 모든 '
 '피보험자 또는 모든 보험수익자의「소득- 45 -세법 시행규칙 별지 제38호 서식에 의한 장애인 증명서의 원본 또는 사본」(이하,「장\n'
 '애인 증명서」라 합니다)을 제출하여 제1조(특별약관의 적용범위) 제1항 제2호에서 정'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000205',
              'chunk_char_len': 291,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
