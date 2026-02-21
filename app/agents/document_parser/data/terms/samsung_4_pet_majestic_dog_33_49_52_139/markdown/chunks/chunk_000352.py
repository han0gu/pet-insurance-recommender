from langchain_core.documents import Document

chunk = Document(
    page_content=('- - 추간판 관련 경막외 신경차단술\n'
 '- - 치, 치수, 치은, 치근, 치조골의 처치\n'
 '※ 본 시술들은 수술의 정의에 해당하지 않는 시술의 예시로, 예시에 기재되어 있지 않다 하더라도\n'
 '수술의 정의에 해당하지 않는 경우 보상되지 않습니다.제4조 (보험금을 지급하지 않는 사유)- 79 -# 회사는 아래의 사유를 원인으로 '
 '하여 생긴 손해는 보상하지 않습니다.- 1. 특별약관 일반사항 제7조(보험금을 지급하지 않는 사유)\n'
 '- 2. 위생관리, 미모를 위한 성형수술. 다만, 사고전 상태로의 회복을 위한 수술은 보상\n'
 '- 합니다.'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
