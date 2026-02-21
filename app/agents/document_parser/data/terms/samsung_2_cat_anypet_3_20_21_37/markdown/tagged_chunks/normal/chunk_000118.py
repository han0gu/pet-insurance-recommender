from langchain_core.documents import Document

chunk = Document(
    page_content=('해지된 것으로 합니다.# 제5조(준용규정)이 추가특별약관에서 정하지 않은 사항은 보통약관 및 해당특별약관을 따릅니다.- 29 -당신에게 '
 '좋은보험 삼성화재# 포괄계약 추가특별약관# (단체계약 특별약관에 적용)# 제1조(적용특칙)- ① 이 추가특별약관(이하 「특별약관」 이라 '
 '합니다)을 첨부한 경우에 보험회사(이하 「회사」 라 합니\n'
 '- 다)와 보험계약자(이하 「계약자」 라 합니다)는 다른 규정에도 불구하고 이 특별약관에 따라 보험\n'
 '- 료를 정산합니다.'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000118',
              'chunk_char_len': 251,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
