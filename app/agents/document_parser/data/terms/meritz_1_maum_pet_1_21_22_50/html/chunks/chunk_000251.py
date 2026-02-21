from langchain_core.documents import Document

chunk = Document(
    page_content=("회사는 피보험자에 대하여 보상책임을 지는 한도 내에서 제1항의 절차에 협조하거나<br>대행합니다.</p><br><h1 id='80' "
 "style='font-size:14px'>【보상책임을 지는 한도】</h1><br><p id='81' "
 "data-category='paragraph' style='font-size:14px'>동일한 사고로 이미 지급한 보험금이나 "
 "가지급보험금이 있는 경우에는 그 금액을 공제<br>한 액수를 말합니다.</p><br><p id='82' "
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
