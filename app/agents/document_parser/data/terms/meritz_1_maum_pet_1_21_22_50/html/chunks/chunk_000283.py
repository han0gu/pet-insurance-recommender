from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 타인을 위<br>한 계약의 경우에는 계약자는 그 타인의 동의를 얻거나 보험증권을 소지한 경우에 한하여<br>계약을 해지할 수 '
 "있습니다.</p><h1 id='116' style='font-size:14px'>제21조(준용규정)</h1><br><p id='117' "
 "data-category='paragraph' style='font-size:14px'>이 특별약관에서 정하지 않은 사항은 보통약관을 "
 "따릅니다.</p><footer id='118' style='font-size:14px'>- 30 -</footer><h1"),
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
