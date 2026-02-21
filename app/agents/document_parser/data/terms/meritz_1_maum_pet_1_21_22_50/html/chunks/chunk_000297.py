from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 보통약관 제16조(계약 후 알릴 의무)에 따라<br>보험료가 변경된 경우에는 예외로 합니다.</p><h1 id='25' "
 "style='font-size:14px'>제3조(준용규정)</h1><br><p id='26' "
 "data-category='paragraph' style='font-size:14px'>이 특별약관에 정하지 않은 사항은 보통약관을 "
 "따릅니다.</p><footer id='27' style='font-size:14px'>- 34 -</footer><h1 id='28' "
 "style='font-size:18px'>보험료"),
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
