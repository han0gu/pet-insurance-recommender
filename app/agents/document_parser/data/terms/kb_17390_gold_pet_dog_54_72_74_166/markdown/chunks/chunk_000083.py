from langchain_core.documents import Document

chunk = Document(
    page_content=('- 5. 보험설계사 등이 계약자 또는 피보험자에게 알릴 기회를 주지 않았거나 계약자\n'
 '- 또는 피보험자가 사실대로 알리는 것을 방해한 경우, 계약자 또는 피보험자에\n'
 '- 게 사실대로 알리지 않게 하였거나 부실한 사항을 알릴 것을 권유했을 때. 다\n'
 '- 만, 보험설계사 등의 행위가 없었다 하더라도 계약자 또는 피보험자가 사실대\n'
 '- 로 알리지 않거나 부실한 사항을 알렸다고 인정되는 경우에는 계약을 해지할\n'
 '- 수 있습니다.\n'
 '- \uf000 제1항에 따라 계약을 해지하였을 때에는 제34조(해약환급금) 제1항에 따른 해약환'),
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
