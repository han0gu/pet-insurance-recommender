from langchain_core.documents import Document

chunk = Document(
    page_content=("id='10' data-category='paragraph' style='font-size:14px'>이 특별약관에 정하지 않은 사항은 "
 "보통약관을 따릅니다.</p><footer id='11' style='font-size:14px'>- 32 -</footer><h1 "
 "id='12' style='font-size:18px'>반려동물 치료비 부보장 특별약관</h1><h1 id='13' "
 "style='font-size:14px'>제1조(보험금을 지급하지 않는 사유)</h1><br><p id='14'"),
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
