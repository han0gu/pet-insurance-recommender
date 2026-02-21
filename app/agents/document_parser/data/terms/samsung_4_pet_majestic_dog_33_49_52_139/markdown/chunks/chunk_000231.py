from langchain_core.documents import Document

chunk = Document(
    page_content=('④ 계약자가 제2항에 따라 보험수익자를 변경하고자 할 경우 계약자와 피보험자가 동일\n'
 '하지 않을 때에는 보험금의 지급사유가 발생하기 전에 피보험자가 서면(「전자서명\n'
 '법」제2조 제2호에 따른 전자서명이 있는 경우로서 상법 시행령 제44조의2에 정하는\n'
 '바에 따라 본인 확인 및 위조ㆍ변조 방지에 대한 신뢰성을 갖춘 전자문서를 포함)으로\n'
 '동의하여야 합니다.\n'
 '⑤ 계약자가 보험수익자를 변경하지 않고 사망한 때에는 계약자 사망시점에 지정되어 있\n'
 '는 보험수익자의 권리가 확정됩니다. 그러나 계약자가 사망한 이후 그 승계인이 보험'),
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
