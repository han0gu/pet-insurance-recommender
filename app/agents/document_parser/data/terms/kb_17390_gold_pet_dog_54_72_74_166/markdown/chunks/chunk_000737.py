from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 회사는 보통약관 제1절 일반조항 제5조(보험금을 지급하지 않는 사유) 및 다음\n'
 '중 어느 한 가지의 경우로 인하여 보험금 지급사유가 발생한 때에는 보험금을 지\n'
 '급하지 않습니다.- 1. 피보험자의 치매를 제외한 정신적 기능장해, 선천성 뇌질환 및 심신상실\n'
 '- 2. 성병\n'
 '- 3. 알코올 중독, 습관성 약품 또는 환각제의 복용 및 사용\n'
 '\uf000 회사는 아래의 의료비로 보험금 지급사유가 발생한 때에는 보험금을 지급하지 않\n'
 '습니다.- 1. 질병을 원인으로 하지 않은 신체검사, 예방접종, 인공유산, 불임시술, 제왕절\n'
 '- 개수술'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
