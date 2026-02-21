from langchain_core.documents import Document

chunk = Document(
    page_content=('서류를 열람할 수 있습니다.# 제6조(준용규정)이 추가특별약관에 정하지 않은 사항은 보통약관 및 해당특별약관을 따릅니다.- 31 '
 '-당신에게 좋은보험 삼성화재# 상품다수구매자단체계약 특별약관# 제1조(적용범위)- ① 이 상품다수구매자단체계약 특별약관(이하 「특별약관」 '
 '이라 합니다)은 단체계약 특별약관 제1조(계\n'
 '- 약의 적용 범위)에도 불구하고 상품판매자가 자기의 관리하에 운영·유지되는 상품의 다수구매자를\n'
 '- 피보험자로 하여 계약을 체결하는 경우에 적용합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000123',
              'chunk_char_len': 258,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
