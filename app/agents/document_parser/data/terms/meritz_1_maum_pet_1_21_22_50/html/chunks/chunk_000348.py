from langchain_core.documents import Document

chunk = Document(
    page_content=(". 피보험자의 3촌 이내의 친족</p><br><p id='18' data-category='paragraph' "
 "style='font-size:14px'>② 제1항에도 불구하고, 지정대리청구인이 지정된 이후에 제1조(적용대상)의 "
 "보험수익자가<br>변경되는 경우에는 이미 지정된 지정대리청구인의 자격은 자동적으로 상실된 것으로<br>봅니다.</p><h1 id='19' "
 "style='font-size:14px'>제4조(지정대리청구인의 변경지정)</h1><br><p id='20' "
 "data-category='paragraph'"),
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
