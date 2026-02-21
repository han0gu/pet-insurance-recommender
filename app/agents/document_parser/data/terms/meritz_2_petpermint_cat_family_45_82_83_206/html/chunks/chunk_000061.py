from langchain_core.documents import Document

chunk = Document(
    page_content=('회사는 보험기간이 끝난 때에 만기환급금(중도인출이 있<br>는 경우에는 중도인출 원금과 이자를 차감하고 적립한 금액<br>을 말합니다)을 '
 '보험수익자에게 지급합니다.<br>\uf000 회사는 계약자 및 보험수익자의 청구에 따라 제1항에 따<br>른 만기환급금을 지급하는 경우 '
 '청구일부터 3영업일 이내에<br>지급합니다.<br>\uf000 회사는 제1항에 따른 만기환급금의 지급시기가 되면 지<br>급시기 7일 '
 '이전에 그 사유와 지급할 금액을 계약자 또는<br>보험수익자에게 알려드리며, 만기환급금을 지급함에 있어<br>지급일까지의 기간에 대한 '
 '이자의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
