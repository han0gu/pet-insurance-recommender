from langchain_core.documents import Document

chunk = Document(
    page_content=('상품다수구매자단체계약 특별약관\n'
 '제1조(적용범위)\n'
 '① 이 상품다수구매자단체계약 특별약관(이하 「특별약관」 이라 합니다)은 단체계약 특별약관 제1조(계 약의 적용 범위)에도 불구하고 '
 '상품판매자가 자기의 관리하에 운영·유지되는 상품의 다수구매자를 피보험자로 하여 계약을 체결하는 경우에 적용합니다. ② 제1항의 상품의 '
 '다수구매자란 각종 재화, 용역 및 서비스의 구매자를 말합니다. ③ 제1항의 단체의 총 피보험자 수는 100인 이상이어야 합니다.\n'
 '제2조(계약자)'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 32},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000153',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
