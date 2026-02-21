from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사는 이 계약의 체결, 유지, 보험금 지급 등을 위하여 위<br>관계 법령에 따라 계약자 및 피보험자의 동의를 받아 다른 '
 '보험회사 및 보험관련단체<br>등에 개인정보를 제공할 수 있습니다.<br>② 회사는 계약과 관련된 개인정보를 안전하게 관리하여야 '
 "합니다.</p><h1 id='6' style='font-size:14px'>제41조(준거법)</h1><br><p id='7' "
 "data-category='paragraph' style='font-size:14px'>이 계약은 대한민국 법에 따라 규율되고 해석되며, "
 '약관에서'),
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
