from langchain_core.documents import Document

chunk = Document(
    page_content=('. 최초계약을 체결한 날부터 3년이 지났을 때<br>4. 보험을 모집한 자(이하 “보험설계사 등”이라 합니다)가 계약자 또는 '
 '피보험자에게<br>알릴 기회를 주지 않았거나 계약자 또는 피보험자가 사실대로 알리는 것을 방해한<br>경우, 계약자 또는 피보험자에게 '
 '사실대로 알리지 않게 하였거나 부실한 사항을 알<br>릴 것을 권유했을 때'),
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
