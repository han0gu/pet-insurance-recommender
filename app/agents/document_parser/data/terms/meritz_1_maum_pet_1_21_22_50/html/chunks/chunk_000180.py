from langchain_core.documents import Document

chunk = Document(
    page_content=("조정)</h1><br><p id='98' data-category='list' style='font-size:14px'>① 계약에 관하여 "
 '분쟁이 있는 경우 분쟁 당사자 또는 기타 이해관계인과 회사는 금융감독<br>원장에게 조정을 신청할 수 있으며, 분쟁조정 과정에서 계약자는 '
 '관계 법령이 정하는<br>바에 따라 회사가 기록 및 유지･관리하는 자료의 열람(사본의 제공 또는 청취를 포함한<br>다)을 요구할 수 '
 '있습니다.<br>② 회사는 일반금융소비자인 계약자가 조정을 통하여 주장하는 권리나 이익의 가액이 ｢금<br>융소비자보호에 관한'),
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
