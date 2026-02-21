from langchain_core.documents import Document

chunk = Document(
    page_content=('- 정에서 정한 지급예정일을 통지한 경우를 포함합니다)에는 그 다음날부터 지급일까지\n'
 '- 의 기간에 대하여 기본계약 약관의 [별표1] 보험금을 지급할 때의 적립이율 계산에서\n'
 '- 정한 이율로 계산한 금액을 보험금에 더하여 지급합니다. 그러나 계약자, 피보험자 또\n'
 '- 는 보험수익자의 책임있는 사유로 지급이 지연된 때에는 그 해당기간에 대한 이자는\n'
 '- 더하여 지급하지 않습니다.\n'
 '- ⑥ 계약자, 피보험자 또는 보험수익자는 제17조(알릴 의무 위반의 효과) 및 제2항의 보험'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
