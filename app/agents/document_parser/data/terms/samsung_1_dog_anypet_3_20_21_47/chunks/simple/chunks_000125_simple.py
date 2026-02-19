from langchain_core.documents import Document

chunk = Document(
    page_content=('제1 조(보상하는 손해)\n'
 '회사는 보통약관 제5조(보상하지 않는 손해) 제2항 제14호에도 불구하고, 피부병(외이염, 면역성 피부병(아토피, 알러지 포함), '
 '세균감염, 곰팡이감염, 기생충 감염, 호르몬성 피부병, 피부트러블을 포함)을 원인으로 하여 생긴 반려동물의 치료비를 보통약관 '
 '제4조(보상하는 손해)에 따라 보상하여 드립니다. 제1항의 피부병 치료비에 대한 회사의 보장은 보험개시일로부터 30일 이내에 발생한 '
 '질병으로 인 한 손해는 보상하여 드리지 않습니다. 단, 이 피부병 보장 특별약관을 갱신하는 경우에는 적용하지 않습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 23},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000125',
              'chunk_char_len': 297,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
