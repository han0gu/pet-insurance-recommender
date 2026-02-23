from langchain_core.documents import Document

chunk = Document(
    page_content=("지정하지 않은 때에는 보험수익자를 피보험자로 합니다.</h1><h1 id='94' "
 "style='font-size:14px'>제14조(대표자의 지정)</h1><br><p id='95' data-category='list' "
 "style='font-size:14px'>① 계약자 또는 보험수익자가 2명 이상인 경우에는 각 대표자를 1명 지정하여야 합니다.<br>이 "
 '경우 그 대표자는 각각 다른 계약자 또는 보험수익자를 대리하는 것으로 합니다.<br>② 지정된 계약자 또는 보험수익자의 소재가 확실하지 '
 '않은 경우에는 이 계약에 관하여'),
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
