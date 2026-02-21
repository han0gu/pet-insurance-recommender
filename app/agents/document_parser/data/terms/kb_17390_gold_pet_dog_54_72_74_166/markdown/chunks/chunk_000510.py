from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사가 청약과 함께 제1회 보험료 등을 받고 청약을 승낙하기 전에 보험금 지급사\n'
 '- 유가 발생하였을 때에도 보장개시일부터 이 특별약관이 정하는 바에 따라 보장을\n'
 '- 합니다.\n'
 '- \uf000 회사는 제2항에도 불구하고 다음 중 한 가지에 해당되는 경우에는 보장을 하지 않\n'
 '- 습니다.\n'
 '- 1. 제7조(계약 전 알릴 의무)의 규정에 따라 계약자 또는 피보험자가 회사에 알\n'
 '- 린 내용이 보험금 지급사유의 발생에 영향을 미쳤음을 회사가 증명하는 경우\n'
 '- 2. 제9조(알릴 의무 위반의 효과)를 준용하여 회사가 보장을 하지 않을 수 있는'),
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
