from langchain_core.documents import Document

chunk = Document(
    page_content=("id='51' style='font-size:14px'>제6조(입원의 정의와 장소)</h1><footer id='52' "
 "style='font-size:14px'>- 5 -</footer><p id='53' data-category='paragraph' "
 "style='font-size:14px'>이 계약에 있어서 「입원」이라 함은 수의사가 상해 또는 질병의 치료가 필요하다고 인정한 "
 '경<br>우로서, 자택 등에서의 치료가 곤란하여 동물병원에 입실하여 수의사의 관리 하에 치료에 전념<br>하는 것을 '
 "말합니다.</p><h1 id='54'"),
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
