from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험설계사 등의 행위가 없었다 하더라도 계약자 또는<br>피보험자가 사실대로 알리지 않거나 부실한 사항을 알렸다고 인정되는 '
 "경우에는 계<br>약을 해지할 수 있습니다.</p><br><p id='101' data-category='list' "
 "style='font-size:14px'>③ 제1항에 따라 계약의 해지가 보험금 지급사유 발생 전에 이루어진 경우, 이로 "
 '인하여<br>회사가 환급하여야 할 보험료가 있을 때에는 보통약관 제33조(보험료의 환급)에 따른<br>보험료를 계약자에게 '
 '지급합니다.<br>④ 제1항 제1호에 따른'),
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
