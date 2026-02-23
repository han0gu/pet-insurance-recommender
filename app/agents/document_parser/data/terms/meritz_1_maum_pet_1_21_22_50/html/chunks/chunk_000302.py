from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>이 특별약관에 정하지 않은 사항은 보통약관을 따릅니다.</p><footer id='39' "
 "style='font-size:14px'>- 35 -</footer><h1 id='40' "
 "style='font-size:18px'>초회보험료자동납입 추가특별약관</h1><p id='41' "
 "data-category='paragraph' style='font-size:14px'>제1조(보험료의 납입)</p><br><p "
 "id='42' data-category='list'"),
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
