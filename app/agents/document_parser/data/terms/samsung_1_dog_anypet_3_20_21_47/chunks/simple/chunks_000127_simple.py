from langchain_core.documents import Document

chunk = Document(
    page_content=('슬관절 수술비용보장 특별약관\n'
 '제1조(보상하는 손해)\n'
 '회사는 보통약관 제5조(보상하지 않는 손해) 제2항 제13호에도 불구하고 슬개골탈구, 십자인대파열, 고 관절탈구(고관절형성부전, '
 '대퇴골두괴사증으로 인한 탈구 포함)를 원인으로 하여 수술을 받은 경우 수술 당일 발생한 수술비 및 치료비를 보상하여 드립니다. 단, '
 '보험개시일로부터 그 날을 포함하여 90일 이내 에 발생한 손해는 보상하여 드리지 않습니다. 이 계약이 갱신계약인 경우에는 적용하지 '
 '않습니다.'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 24},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000127',
              'chunk_char_len': 253,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
