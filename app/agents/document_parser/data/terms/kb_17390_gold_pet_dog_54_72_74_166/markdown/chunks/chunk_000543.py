from langchain_core.documents import Document

chunk = Document(
    page_content=('질병(이하 사고라 합니다)이 발생하여 그 치료를 직접적인 목적으로 국내에서 수\n'
 '의사에게 치료를 받은 때에는 1일당 피보험자가 부담한 반려동물의 치료에 사용된\n'
 '비용(각종 할인 및 감면, 사후환급금액 등을 제외한 실수납액을 의미합니다. 이하\n'
 '의료비라 합니다)을 제2항에 따라 이 특별약관의 보험가입금액을 한도로 보험수익\n'
 '상\n'
 '자에게 반려동물의료비보험금(이하 의료비보험금이라 합니다)으로 보상하여 드립\n'
 '해\n'
 '니다. 단, 보험기간 중에 발생한 사고로 회사가 지급하는 연간 의료비보험금의 총'),
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
