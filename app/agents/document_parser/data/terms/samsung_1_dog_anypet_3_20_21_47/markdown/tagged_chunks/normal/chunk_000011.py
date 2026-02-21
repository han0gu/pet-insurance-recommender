from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5 -당신에게 좋은보험 삼성화재# 제5조(보상하지 않는 손해)# ① 회사는 아래의 사유로 인한 손해는 보상하여 드리지 않습니다.- '
 '1. 계약자 및 피보험자의 고의, 중대한 과실\n'
 '- 2. 지진, 분화, 풍수해 및 이와 유사한 자연재해로 생긴 손해\n'
 '- 3. 전쟁, 혁명, 내란, 폭동, 소요 기타 유사한 사태로 생긴 손해\n'
 '- 4. 피보험자의 질병, 심신상실 또는 정신질환으로 인한 손해\n'
 '- 5. 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의 방사성, 폭발성, 그 밖의 유해한 특성 또'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000011',
              'chunk_char_len': 272,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
