from langchain_core.documents import Document

chunk = Document(
    page_content=(". 단, 흡인, 천자 등의 조치, 신경(神經)차단(NERVE BLOCK), 미용성형</p><footer id='33' "
 "style='font-size:18px'>- 3 -</footer><p id='34' data-category='paragraph' "
 "style='font-size:14px'>목적의 수술, 피임목적의 수술 및 검사, 진단을 위한 수술(생검, 복강경검사 등)은 "
 "제외합니<br>다.</p><br><h1 id='35' style='font-size:14px'>【 용어의 정의 】</h1><br><p "
 "id='36'"),
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
