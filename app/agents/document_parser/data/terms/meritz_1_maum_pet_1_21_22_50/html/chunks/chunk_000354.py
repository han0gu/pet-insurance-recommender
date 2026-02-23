from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 2인의 지정대리청구인이 지정된 경우에는 그<br>중 대표대리인이 보험금을 청구하고 수령할 수 있으며, 대표대리인이 사망 등의 '
 '사유로<br>보험금 청구가 불가능한 경우에는 대표가 아닌 지정대리청구인도 보험금을 청구하고 수<br>령할 수 있습니다.<br>② 회사가 '
 '보험금을 지정대리청구인에게 지급한 경우에는 그 이후 보험금 청구를 받더라도<br>회사는 이를 지급하지 않습니다.</p><h1 '
 "id='26' style='font-size:14px'>제6조(보험금의 청구)</h1><br><p id='27'"),
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
