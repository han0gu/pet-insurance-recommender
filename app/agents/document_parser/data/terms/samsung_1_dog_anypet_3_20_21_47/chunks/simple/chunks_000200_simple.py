from langchain_core.documents import Document

chunk = Document(
    page_content=('제5조(보험료의 환급)\n'
 '계약자의 책임있는 사유로 계약을 해지하는 경우에는 보통약관 제30조(보험료의 환급)의 규정에도 불 구하고 이미 경과한 기간에 대하여 '
 '단기요율(1년 미만의 기간에 적용되는 요율)로 계산한 보험료를 뺀 잔액을 돌려드립니다.\n'
 '제6조(준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 40},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000200',
              'chunk_char_len': 176,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
