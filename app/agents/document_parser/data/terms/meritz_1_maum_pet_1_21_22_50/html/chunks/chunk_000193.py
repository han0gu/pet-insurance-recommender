from langchain_core.documents import Document

chunk = Document(
    page_content=("하지 않을 정도로 현저하게 공정성을<br>잃은 것을 말합니다.</p><h1 id='4' "
 "style='font-size:14px'>제40조(개인정보보호)</h1><br><p id='5' data-category='list' "
 "style='font-size:14px'>① 회사는 이 계약과 관련된 개인정보를 이 계약의 체결, 유지, 보험금 지급 등을 "
 '위하여<br>「개인정보 보호법」,「신용정보의 이용 및 보호에 관한 법률」 등 관계 법령에 정한<br>경우를 제외하고 계약자, 피보험자 '
 '또는 보험수익자의 동의없이 수집, 이용, 조회'),
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
