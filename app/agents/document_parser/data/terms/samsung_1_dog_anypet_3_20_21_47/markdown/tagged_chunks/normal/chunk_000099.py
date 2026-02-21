from langchain_core.documents import Document

chunk = Document(
    page_content=('기재된 총보상횟수를 한도로 합니다.# 제3조(준용규정)이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.당신에게 좋은보험 삼성화재- '
 '22 -# 피부병 보장 특별약관# 제1 조(보상하는 손해)- 회사는 보통약관 제5조(보상하지 않는 손해) 제2항 제14호에도 불구하고, '
 '피부병(외이염, 면역성\n'
 '- 피부병(아토피, 알러지 포함), 세균감염, 곰팡이감염, 기생충 감염, 호르몬성 피부병, 피부트러블을\n'
 '- 포함)을 원인으로 하여 생긴 반려동물의 치료비를 보통약관 제4조(보상하는 손해)에 따라 보상하여\n'
 '- 드립니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'limit', 'risk_domains': ['digestive', 'skin']},
 'indexing': {'chunk_id': 'chunk_000099',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
