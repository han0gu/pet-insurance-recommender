from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2. 피보험자의 치매를 제외한 정신적 기능장해, 선천성 뇌질환 및 심신상실\n'
 '- 3. 성병\n'
 '- 4. 알콜중독, 습관성 약품 또는 환각제의 복용 및 사용\n'
 '# ② 회사는 아래의 사유로 생긴 손해는 보상하지 않습니다.- 1. 질병을 원인으로 하지 않는 신체검사, 예방접종, 인공유산, 불임시술, '
 '제왕절개수술\n'
 '- 2. 피로, 권태, 심신허약 등을 치료하기 위한 안정치료\n'
 '# 제7조 (특별약관의 소멸)피보험자 또는 보험증권에 기재된 반려견이 보험기간 중에 사망하였을 경우에는 "보험료'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000528',
              'chunk_char_len': 266,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
