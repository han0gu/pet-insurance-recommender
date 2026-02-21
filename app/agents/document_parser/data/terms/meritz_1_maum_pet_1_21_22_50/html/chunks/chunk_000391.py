from langchain_core.documents import Document

chunk = Document(
    page_content=("따릅니다.<br>② 소득세법 등 관련법규가 제·개정 또는 폐지되는 경우 변경된 법령을 따릅니다.</p><footer id='80' "
 "style='font-size:14px'>- 47 -</footer><caption id='81' "
 "style='font-size:14px'><부표1> 보험금을 지급할 때의 적립이율 계산</caption><p id='82' "
 "data-category='paragraph' style='font-size:14px'>(보통약관 제9조 제4항 관련)</p><table "
 "id='83'"),
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
