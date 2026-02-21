from langchain_core.documents import Document

chunk = Document(
    page_content=('# 제7조(보험기간의 설정)회사는 새로이 증가 또는 교체되는 피보험자의 보험기간은 계약자가 요청하는 기간으로 합니다. 다만,\n'
 '이 계약기간 중 피보험자 감소의 경우 당해 피보험자의 계약은 해지된 것으로 합니다.# 제8조(적용특칙)회사는 계약자에게만 보험증권을 '
 '드립니다.제9조(준용규정)이 추가특별약관에 정하지 않은 사항은 보통약관 및 해당특별약관을 따릅니다.[양식1]| 피보험자명 | 주민등록번호 '
 '(필요시) | 주소 (필요시) | 전화번호 (필요시) | 상품구입일 | 날인 |'),
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
 'indexing': {'chunk_id': 'chunk_000134',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
