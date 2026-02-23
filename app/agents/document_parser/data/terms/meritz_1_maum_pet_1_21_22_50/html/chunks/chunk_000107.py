from langchain_core.documents import Document

chunk = Document(
    page_content=('승낙으로 이루어집니다.<br>② 회사는 피보험자가 계약에 적합하지 않은 경우에는 승낙을 거절하거나 별도의 조건(보<br>험가입금액 제한, '
 '일부보장 제외, 보험금 삭감, 보험료 할증 등)을 붙여 승낙할 수 있습<br>니다.<br>③ 회사는 계약의 청약을 받고, 제1회 보험료를 '
 '받은 경우에 건강진단을 받지 않는 계약은<br>청약일, 진단계약은 진단일(재진단의 경우에는 최종 진단일)부터 30일 이내에 승낙 '
 '또<br>는 거절하여야 하며, 승낙한 때에는 보험증권을 드립니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
