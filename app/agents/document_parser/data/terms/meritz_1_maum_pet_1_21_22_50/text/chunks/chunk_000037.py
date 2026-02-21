from langchain_core.documents import Document

chunk = Document(
    page_content=('금의 일부를 먼저 지급하는 제도로 피보험자가 필요로 하는 비용을 보전해 주기 위\n'
 '해 회사가 먼저 지급하는 임시 교부금을 말합니다.④ 회사는 제1항의 규정에 정한 지급기일내에 보험금을 지급하지 않았을 때(제2항의 규정\n'
 '에서 정한 지급예정일을 통지한 경우를 포함합니다)에는 그 다음날부터 지급일까지의\n'
 '기간에 대하여 <부표1> ‘보험금을 지급할 때의 적립이율 계산’에서 정한 이율로 계산\n'
 '한 금액을 보험금에 더하여 지급합니다. 그러나 계약자, 피보험자 또는 보험수익자의'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
