from langchain_core.documents import Document

chunk = Document(
    page_content=("계약자는 모든 피보험자 또는 모든 보험수익자의「소득</p><footer id='63' style='font-size:14px'>- 45 "
 "-</footer><p id='64' data-category='paragraph' style='font-size:14px'>세법 "
 '시행규칙 별지 제38호 서식에 의한 장애인 증명서의 원본 또는 사본」(이하,「장<br>애인 증명서」라 합니다)을 제출하여 '
 "제1조(특별약관의 적용범위) 제1항 제2호에서 정<br>한 조건에 해당함을 회사에 알려야 합니다.</p><br><p id='65'"),
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
 'indexing': {'chunk_id': 'chunk_000378',
              'chunk_char_len': 289,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
