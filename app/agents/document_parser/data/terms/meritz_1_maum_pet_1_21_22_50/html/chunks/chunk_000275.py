from langchain_core.documents import Document

chunk = Document(
    page_content=('. 회사는 전자문<br>서가 수신되지 않은 것을 확인한 경우에는 서면(등기우편 등)으로 다시 알려드립니다.</p><br><p '
 "id='104' data-category='list' style='font-size:14px'>⑤ 손해가 제1항 제1호 또는 제2호에 "
 '해당되는 사실로 생긴 것이 아님을 계약자 또는 피<br>보험자가 증명한 경우에는 제4항에 관계없이 보상합니다.<br>⑥ 회사는 다른 '
 '보험가입내역에 대한 계약 전․후 알릴 의무 위반을 이유로 계약을 해지하<br>거나 보험금 지급을 거절하지 않습니다.<br>⑦ 보통약관'),
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
