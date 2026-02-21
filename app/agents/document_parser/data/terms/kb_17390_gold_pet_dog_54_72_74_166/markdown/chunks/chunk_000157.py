from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 제1항 및 제2항에 따른 보험료의 자동대출납입 기간은 최초 자동대출납입일부터 1\n'
 '- 년을 한도로 하며 그 이후의 기간에 대한 보험료의 자동대출납입을 위해서는 제1항\n'
 '- 에 따라 재신청을 하여야 합니다.\n'
 '- \uf000 보험료의 자동대출 납입이 행하여진 경우에도 자동대출 납입전 납입최고(독촉)기\n'
 '- 간이 끝나는 날의 다음날부터 1개월 이내에 계약자가 계약의 해지를 청구한 때에는\n'
 '- 회사는 보험료의 자동대출 납입이 없었던 것으로 하여 제34조(해약환급금) 제1항\n'
 '- 에 따른 해약환급금을 지급합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
