from langchain_core.documents import Document

chunk = Document(
    page_content=('보험<br>료가 정산되기 이전일지라도 새로이 증가 또는 교체된 보험의 목적에 대해 생긴 손해<br>를 보상합니다.</p><h1 '
 "id='79' style='font-size:14px'>제2조(보험의 목적의 명부)</h1><br><p id='80' "
 "data-category='paragraph' style='font-size:14px'>계약자는 항상 보험의 목적의 명부를 비치하여 "
 "회사가 열람을 요구할 경우에는 이에 따라<br>야 합니다.</p><h1 id='81'"),
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
