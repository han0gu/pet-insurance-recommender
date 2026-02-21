from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 이미 보<br>험금 지급사유가 발생한 경우에는 보험금 지급에 영향을 미치지 않습니다.</p><br><p id='86' "
 "data-category='paragraph' style='font-size:14px'>② 회사가 제1항에 따라 계약을 해지한 경우 "
 '회사는 그 취지를 계약자에게 통지하고, 해지<br>시 회사가 환급하여야 할 보험료가 있을 경우에는 제33조(보험료의 환급)에 따른 '
 "보험<br>료를 계약자에게 지급합니다.</p><h1 id='87' style='font-size:14px'>제32조(회사의 파산선고와"),
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
