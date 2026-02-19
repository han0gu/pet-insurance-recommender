from langchain_core.documents import Document

chunk = Document(
    page_content=('세법 시행규칙 별지 제38호 서식에 의한 장애인 증명서의 원본 또는 사본」(이하,「장 애인 증명서」라 합니다)을 제출하여 '
 '제1조(특별약관의 적용범위) 제1항 제2호에서 정 한 조건에 해당함을 회사에 알려야 합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 46},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000246',
              'chunk_char_len': 119,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.9}},
)
