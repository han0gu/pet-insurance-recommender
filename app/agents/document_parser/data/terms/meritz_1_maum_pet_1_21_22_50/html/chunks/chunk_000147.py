from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 해지 전에 발생한 보험금 지급사유<br>에 대하여 회사는 보상하여 드립니다.</p><br><p id='55' "
 "data-category='list' style='font-size:14px'>1. 계약자(보험수익자와 계약자가 다른 경우 보험수익자를 "
 '포함합니다)에게 납입최고<br>(독촉)기간 내에 연체보험료를 납입하여야 한다는 내용<br>2'),
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
