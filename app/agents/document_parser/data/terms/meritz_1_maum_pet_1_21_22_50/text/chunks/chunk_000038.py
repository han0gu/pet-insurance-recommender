from langchain_core.documents import Document

chunk = Document(
    page_content=('한 금액을 보험금에 더하여 지급합니다. 그러나 계약자, 피보험자 또는 보험수익자의\n'
 '책임 있는 사유로 지급이 지연된 때에는 그 해당기간에 대한 이자는 더하여 지급하지\n'
 '않습니다.\n'
 '⑤ 계약자, 피보험자 또는 보험수익자는 제2항의 보험금 지급사유조사와 관련하여 동물병\n'
 '원 등 의료기관, 경찰서 등 관공서에 대한 회사의 서면에 의한 조사요청에 동의하여야\n'
 '합니다. 다만, 정당한 사유 없이 이에 동의하지 않을 경우 사실 확인이 끝날 때까지 회\n'
 '사는 보험금 지급지연에 따른 이자를 지급하지 않습니다.'),
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
