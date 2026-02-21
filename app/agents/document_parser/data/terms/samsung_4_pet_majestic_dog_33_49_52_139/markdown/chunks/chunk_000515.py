from langchain_core.documents import Document

chunk = Document(
    page_content=('계약일 : 2022년 4월 10일 ⇒ 계약해당일 : 매년 4월 10일\n'
 '단 , 계약해당일 2월 29일이 없을 경우에는 2월 28일을 계약해당일로 합니다.# 제19조 (특별약관의 소멸)① 보험증권에 기재된 '
 '반려견이 보험기간 중에 사망하여 보험의 목적에 대해 이 특별약\n'
 '관에서 정한 보험금 지급사유가 더이상 발생할 수 없는 경우에는 “보험료 및 해약환\n'
 '급금 산출방법서”에 정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약\n'
 '자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없\n'
 '습니다.'),
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
