from langchain_core.documents import Document

chunk = Document(
    page_content=("id='74' data-category='paragraph' style='font-size:14px'>이 특별약관에 정하지 않은 사항은 "
 "보통약관을 따릅니다.</p><footer id='75' style='font-size:14px'>- 38 -</footer><h1 "
 "id='76' style='font-size:18px'>단체계약 보험료정산 추가특별약관</h1><h1 id='77' "
 "style='font-size:14px'>제1조(보험료의 정산)</h1><br><p id='78' data-category='list'"),
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
