from langchain_core.documents import Document

chunk = Document(
    page_content=('잔액을 돌려드립니다.# 제6조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 40 -당신에게 좋은보험 삼성화재# '
 '상품다수구매자단체계약 보험료정산 추가특별약관# (상품다수구매자단체계약 특별약관에 적용)# 제1조(보험료의 정산)이 보험료정산 특별약관 '
 '(이하 「특별약관」 이라 합니다)은 상품다수구매자단체계약 특별약관 제4조(피'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000162',
              'chunk_char_len': 191,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
