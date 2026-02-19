from langchain_core.documents import Document

chunk = Document(
    page_content=('상품다수구매자단체계약 보험료정산 추가특별약관\n'
 '(상품다수구매자단체계약 특별약관에 적용)\n'
 '제1조(보험료의 정산)\n'
 '이 보험료정산 특별약관 (이하 「특별약관」 이라 합니다)은 상품다수구매자단체계약 특별약관 제4조(피 보험자의 증가, 감소 또는 교체) '
 '제2항에도 불구하고 이 특별약관에 따라 보험료를 정산합니다.\n'
 '제2조(보험가입금액)\n'
 '상품다수구매자단체계약 특별약관 제3조(보험가입금액)의 규정에 관계없이 계약자가 피보험자의 보험가 입금액을 각기 달리하여 가입하고자 할 '
 '경우에 회사는 계약사항을 고려하여 이를 승인할 수 있습니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 33},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000158',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
