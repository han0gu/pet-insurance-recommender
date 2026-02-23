from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 회사<br>의 고의 또는 과실로 계약이 무효로 된 경우와 회사가 승낙 전에 무효임을 알았거나 알 수<br>있었음에도 불구하고 '
 '보험료를 반환하지 않은 경우에는 보험료를 납입한 날의 다음날부터<br>반환일까지의 기간에 대하여 회사는 보험개발원이 공시하는 '
 "보험계약대출이율을 연단위 복<br>리로 계산한 금액을 더하여 돌려 드립니다.</p><h1 id='107' "
 "style='font-size:14px'>제18조(타인을 위한 계약)</h1><br><p id='108' "
 "data-category='list'"),
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
