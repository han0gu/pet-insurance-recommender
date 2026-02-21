from langchain_core.documents import Document

chunk = Document(
    page_content=('# 납입을 재촉하는 것을 말합니다.- 제17조(보험료의 납입을 연체하여 해지된 특별약관의 부활(효력회복))\n'
 '- \uf000 제16조(보험료의 납입이 연체되는 경우 납입최고(독촉)와 특별약관의 해지)에 따\n'
 '- 라 특별약관이 해지되었으나 해약환급금을 받지 않은 경우(보험계약대출 등에 따\n'
 '- 라 해약환급금이 차감되었으나 받지 않은 경우 또는 해약환급금이 없는 경우를 포\n'
 '- 함합니다) 계약자는 해지된 날부터 3년 이내에 회사가 정한 절차에 따라 특별약관\n'
 '- 의 부활(효력회복)을 청약할 수 있습니다. 회사가 부활(효력회복)을 승낙한 때에'),
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
