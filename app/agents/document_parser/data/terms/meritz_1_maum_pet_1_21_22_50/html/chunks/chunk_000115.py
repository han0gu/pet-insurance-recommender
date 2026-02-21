from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 계약자가 제1회 보험료 등을 신용카드로 납입한 계약의 청약을 철회하는 경<br>우에는 회사는 청약의 철회를 접수한 날부터 '
 '3영업일 이내에 해당 신용카드회사로 하<br>여금 대금청구를 하지 않도록 해야 하며, 이 경우 회사는 보험료를 반환한 것으로 '
 "봅니<br>다.</p><br><p id='15' data-category='list' style='font-size:14px'>⑤ 청약을 "
 '철회할 때에 이미 보험금 지급사유가 발생하였으나 계약자가 그 보험금 지급사<br>유가 발생한 사실을 알지 못한 경우에는 청약철회의 효력은'),
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
