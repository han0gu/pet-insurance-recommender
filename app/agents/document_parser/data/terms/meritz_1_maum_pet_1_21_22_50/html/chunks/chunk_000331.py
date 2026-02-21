from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제1호에도 불구하고 보험의 목적의 정보의 변경에 관한 서류 제출 시기는 계약자와 별<br>도로 협의하여 변경할 수 '
 "있습니다.</p><h1 id='86' style='font-size:14px'>제5조(준용규정)</h1><br><p id='87' "
 "data-category='paragraph' style='font-size:14px'>이 추가특별약관에 정하지 않은 사항은 보통약관 및 "
 "단체계약 특별약관을 따릅니다.</p><footer id='88' style='font-size:14px'>- 39 "
 '-</footer><h1'),
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
