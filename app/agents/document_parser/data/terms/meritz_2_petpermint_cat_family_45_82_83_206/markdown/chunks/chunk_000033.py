from langchain_core.documents import Document

chunk = Document(
    page_content=('립됩니다.# 【운용자산이익률】직전 1년간의 운용자산에 대한 투자영업수익과 투자영\n'
 '업비용 등을 고려하여 산출# 【외부지표금리】국고채, 회사채, 통화안정증권, 양도성예금증서 등을\n'
 '고려하여 산출# 제10조(만기환급금의 지급)\uf000 회사는 보험기간이 끝난 때에 만기환급금(중도인출이 있\n'
 '는 경우에는 중도인출 원금과 이자를 차감하고 적립한 금액\n'
 '을 말합니다)을 보험수익자에게 지급합니다.\n'
 '\uf000 회사는 계약자 및 보험수익자의 청구에 따라 제1항에 따\n'
 '른 만기환급금을 지급하는 경우 청구일부터 3영업일 이내에\n'
 '지급합니다.'),
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
