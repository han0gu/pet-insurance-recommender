from langchain_core.documents import Document

chunk = Document(
    page_content=('지 당시의 계약자적립액 및 미경과보험료를 반환하여 드립니다.- 제35조(보험계약대출)\n'
 '- \uf000 계약자는 이 계약의 해약환급금 범위 내에서 회사가 정한 방법에 따라 대출(이하 "\n'
 '- 보험계약대출"이라 합니다)을 받을 수 있습니다. 그러나, 순수보장성보험 등 보험\n'
 '- 상품의 종류에 따라 보험계약대출이 제한될 수도 있습니다.\n'
 '- \uf000 계약자는 제1항에 따른 보험계약대출금과 그 이자를 언제든지 상환할 수 있으며 상\n'
 '- 환하지 않은 때에는 회사는 보험금, 해약환급금 등의 지급사유가 발생한 날에 지급'),
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
