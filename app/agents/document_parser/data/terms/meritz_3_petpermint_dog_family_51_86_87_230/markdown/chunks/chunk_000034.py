from langchain_core.documents import Document

chunk = Document(
    page_content=('른 만기환급금을 지급하는 경우 청구일부터 3영업일 이내에\n'
 '지급합니다.\n'
 '\uf000 회사는 제1항에 따른 만기환급금의 지급시기가 되면 지\n'
 '급시기 7일 이전에 그 사유와 지급할 금액을 계약자 또는\n'
 '보험수익자에게 알려드리며, 만기환급금을 지급함에 있어\n'
 '지급일까지의 기간에 대한 이자의 계산은【별표1(보험금을\n'
 '지급할 때의 적립이율 계산)】에 따릅니다.\n'
 '\uf000 보험료 납입기간 중에 제2조(용어의 정의)에서 정한 적\n'
 '립보험료를 감액하거나 중도인출을 하는 경우 제1항의 만기'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
