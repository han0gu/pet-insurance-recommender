from langchain_core.documents import Document

chunk = Document(
    page_content=('치료비는 보상하여 드립니다. 다만, 사고일 또는 발병일부터 365일이내의 치료인 경우에 한합니다.# 제5조(보상하지 않는 손해)# ① '
 '회사는 아래의 사유를 원인으로 하여 생긴 손해는 보상하지 않습니다.- 1. 계약자, 피보험자, 이들의 가족 또는 사용인의 고의 또는 '
 '중대한 과실\n'
 '- 2. 지진, 분화, 해일, 홍수 또는 이와 유사한 자연재해로 생긴 손해\n'
 '- 3. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소요, 기타 이들과 유사한 사태'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000011',
              'chunk_char_len': 250,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
