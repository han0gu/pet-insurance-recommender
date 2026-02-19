from langchain_core.documents import Document

chunk = Document(
    page_content=('① 회사는 아래의 사유를 원인으로 하여 생긴 손해는 보상하지 않습니다.\n'
 '1. 보통약관 제5조 (보험금을 지급하지 않는 사유) 제1항 2. 피보험자의 치매를 제외한 정신적 기능장해, 선천성 뇌질환 및 심신상실 '
 '3. 성병 4. 알콜중독, 습관성 약품 또는 환각제의 복용 및 사용\n'
 '② 회사는 아래의 사유로 생긴 손해는 보상하지 않습니다.\n'
 '1. 질병을 원인으로 하지 않는 신체검사, 예방접종, 인공유산, 불임시술, 제왕절개수술 2. 피로, 권태, 심신허약 등을 치료하기 위한 '
 '안정치료\n'
 '제7조 (특별약관의 소멸)'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 67,
         'page': 94},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000610',
              'chunk_char_len': 281,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
