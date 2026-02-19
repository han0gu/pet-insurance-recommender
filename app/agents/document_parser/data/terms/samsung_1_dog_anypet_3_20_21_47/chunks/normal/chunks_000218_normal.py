from langchain_core.documents import Document

chunk = Document(
    page_content=('① 이 특약에 가입하고자 하는 계약자는 모든 피보험자 또는 모든 보험수익자의 「소득세법 시행규칙 별지 제38호 서식에 의한 장애인증명서의 '
 '원본 또는 사본」 (이하, "장애인증명서"라 합니다)을 제 출하여 제1조(특약의 적용범위) 제1항 제2호에서 정한 조건에 해당함을 회사에 '
 '알려야 합니다'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 44},
 'term_type': 'special',
 'clause': {'clause_type': 'claim', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000218',
              'chunk_char_len': 161,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
