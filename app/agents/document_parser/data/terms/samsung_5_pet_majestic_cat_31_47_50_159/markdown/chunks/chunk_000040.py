from langchain_core.documents import Document

chunk = Document(
    page_content=('- 경우 청구일부터 3영업일 이내에 지급합니다.\n'
 '- ③ 회사는 제1항에 의한 만기환급금의 지급시기가 되면 지급시기 7일 이전에 그 사유와\n'
 '- 지급할 금액을 계약자 또는 보험수익자에게 알려드리며, 만기환급금을 지급함에 있어\n'
 '- 지급일까지의 기간에 대한 이자의 계산은 보험금을 지급할 때의 적립이율 계산([별표\n'
 '- 1] 보험금을 지급할 때의 적립이율 계산 참조)에 따릅니다.\n'
 '- ④ 제10조(환급금의 중도인출) 제1항에 따라 환급금을 중도인출한 경우에는 중도인출금'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
