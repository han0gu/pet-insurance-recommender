from langchain_core.documents import Document

chunk = Document(
    page_content=('입이 연체되는 경우 납입최고(독촉)와 계약의 해지)에 따라\n'
 '계약이 해지되는 때에는 즉시 해약환급금에서 보험계약대출\n'
 '의 원금과 이자를 차감합니다.\n'
 '\uf000 회사는 보험수익자에게 보험계약대출 사실을 통지할 수\n'
 '있습니다.# 제37조(배당금의 지급)회사는 이 계약에 대하여 계약자에게 배당금을 지급하지 않\n'
 '습니다.# 제38조(중도인출)\uf000 계약자는 보장개시일부터 2년 이상 지난 유효한 계약으\n'
 '로서 계약자의 요청이 있는 경우에 한하여 보험연도 기준\n'
 '연4회에 한하여 중도인출 할 수 있습니다.\uf000 제1항의 중도인출금은 계약자가 요청한 시점에서 계산된'),
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
