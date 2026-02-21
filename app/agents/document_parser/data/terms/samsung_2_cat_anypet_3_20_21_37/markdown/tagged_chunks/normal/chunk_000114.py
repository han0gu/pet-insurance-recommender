from langchain_core.documents import Document

chunk = Document(
    page_content=('의 권리를 행사할 수 있습니다.# 제8조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.- 28 -당신에게 좋은보험 '
 '삼성화재# 보험료정산 추가특별약관# (단체계약 특별약관에 적용)# 제1조(보험료의 정산)- 회사는 단체계약 특별약관 제4조(보험의 목적의 '
 '증가 감소 또는 교체) 제2항에도 불구하고 이 추가\n'
 '- 특별약관에 따라 보험료를 정산합니다.\n'
 '- ② 회사는 단체계약 특별약관 제4조(보험의 목적의 증가 감소 또는 교체) 제3항과 관계없이 보험료가 정'),
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
 'indexing': {'chunk_id': 'chunk_000114',
              'chunk_char_len': 262,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
