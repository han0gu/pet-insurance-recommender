from langchain_core.documents import Document

chunk = Document(
    page_content=("목적의 보험기간은 계약자가<br>요청하는 기간으로 합니다.</p><h1 id='5' "
 "style='font-size:14px'>제3조(보험료의 납입)</h1><br><p id='6' data-category='list' "
 "style='font-size:14px'>① 계약자는 새로이 증가된 보험의 목적에 대하여 일단위로 계산된 추가보험료를 납입하여<br>야 "
 '합니다.<br>② 새로이 증가된 보험의 목적의 보험기간이 시작된 후라도 다른 약정이 없으면 추가 보험<br>료를 받기 전에 생긴 손해는 '
 '보상하여 드리지 않습니다.</p><h1'),
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
