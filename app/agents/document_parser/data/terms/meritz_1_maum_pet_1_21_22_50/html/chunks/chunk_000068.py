from langchain_core.documents import Document

chunk = Document(
    page_content=("계산)</h1><br><p id='75' data-category='paragraph' style='font-size:14px'>① "
 '동일한 반려동물과 동일한 사고에 관하여 보험금을 지급하는 다른 계약(공제계약을 포<br>함합니다)이 있을 경우 각 계약에 대하여 다른 '
 '계약이 없는 것으로 하여 각각 산출한<br>지급보험금의 합계액이 피보험자가 부담한 비용금액을 초과할 때에는 아래에 따라 보<br>험금을 '
 "지급합니다.</p><h1 id='76' style='font-size:14px'>피보험자가 부담한 총 비용금액</h1><br><p"),
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
