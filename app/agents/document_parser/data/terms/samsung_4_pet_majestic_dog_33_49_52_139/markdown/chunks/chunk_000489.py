from langchain_core.documents import Document

chunk = Document(
    page_content=('[장해지급률]\n'
 '질병이나 상해에 대하여 치유 후 남아있는 영구적인 장해에 의한 신체의 노동력 상실정도를 %로\n'
 '나타낸 것을 말합니다.④ 회사는 제1항의 규정에 정한 지급기일 내에 보험금을 지급하지 않았을 때(제2항의 규\n'
 '정에서 정한 지급예정일을 통지한 경우를 포함합니다)에는 그 다음날부터 지급일까지\n'
 '의 기간에 대하여 보험금을 지급할 때의 적립이율 계산([별표1] 참조)에서 정한 이율- 102 -로 계산한 금액을 보험금에 더하여 '
 '지급합니다. 그러나 계약자, 피보험자 또는 보험수'),
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
