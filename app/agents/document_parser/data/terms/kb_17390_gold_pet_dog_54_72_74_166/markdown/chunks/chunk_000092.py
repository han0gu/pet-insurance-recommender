from langchain_core.documents import Document

chunk = Document(
    page_content=('- 돌려드립니다.\n'
 '통약관제4 관 보험계약의 성립과 유지제18조(보험계약의 성립)별표- \uf000 계약은 계약자의 청약과 회사의 승낙으로 이루어집니다.\n'
 '- \uf000 회사는 보험의 목적 및 피보험자가 계약에 적합하지 않은 경우에는 승낙을 거절하\n'
 '- 거나 별도의 조건(보험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증 법\n'
 '- 등)을 붙여 승낙할 수 있습니다. ㆍ\n'
 '- \uf000 회사는 계약의 청약을 받고, 제1회 보험료를 받은 경우에 건강진단을 받지 않는 계 규정\n'
 '- 약은 청약일, 진단계약은 진단일(재진단의 경우에는 최종 진단일)부터 30일 이내'),
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
