from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1. 계약자에게 전자적 상품설명장치를 활용하여 제1항에 따른 납입최고(독촉) 등\n'
 '- 을 한다는 사실을 미리 안내하고 동의를 받을 것\n'
 '- 2. 전자적 상품설명장치를 활용하여 안내한 납입최고(독촉) 등을 계약자가 모두\n'
 '- 수신하고 이해하였음을 확인할 것\n'
 '- 3. 계약자가 질의를 하거나 추가적인 설명을 요청하는 등 전자적 상품설명장치의\n'
 '- 활용을 중단할 것을 요구하는 경우, 회사는 전화(음성녹음) 방법으로 전환하여\n'
 '- 제1항에 따른 납입최고(독촉) 등을 실시할 것'),
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
