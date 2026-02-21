from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한 이 경우<br>계약 해지로 인하여 회사가 환급하여야 할 보험료가 있을 때에는 보통약관 제33조(보<br>험료의 환급)에 따른 '
 '보험료를 계약자에게 지급합니다. 회사가 전자문서로 안내하고자<br>할 경우에는 계약자에게 서면 또는 「전자서명법」 제2조 제2호에 따른 '
 '전자서명으로<br>동의를 얻어 수신확인을 조건으로 전자문서를 송신하여야 합니다. 계약자의 전자문서<br>수신이 확인되기 전까지는 그 '
 '전자문서는 송신되지 않은 것으로 봅니다'),
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
