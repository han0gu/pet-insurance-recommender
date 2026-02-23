from langchain_core.documents import Document

chunk = Document(
    page_content=('보험금이 지급되었다면 이전의 후유장해에 해당하는 보험금- \n'
 '※ 보험금 지급사유에 해당되지 않은 경우란 장해의 원인이 보장개시 이전에 발생했거나 약관상\n'
 '보험금을 지급하지 않는 사유에 해당하는 경우 등을 말합니다.⑧ 회사가 지급하여야 할 하나의 상해로 인한 상해 후유장해보험금은 상해 '
 '후유장해보험\n'
 '가입금액을 한도로 합니다.- 69 -# 제 3조 (특별약관의 소멸)피보험자가 보험기간 중에 사망하였을 경우에는 "보험료 및 해약환급금 '
 '산출방법서"에서\n'
 '정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약자적립액 및 미경과보험료'),
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
