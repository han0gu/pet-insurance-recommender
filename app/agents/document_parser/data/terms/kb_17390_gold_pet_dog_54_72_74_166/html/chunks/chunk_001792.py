from langchain_core.documents import Document

chunk = Document(
    page_content=('[B58.3+: 폐 톡소포자충증(J17.3*)]</td><td>B58.3+</td></tr><tr><td>상세불명 병원체의 '
 "폐렴</td><td>J18</td></tr></tbody></table><br><table id='109' "
 "style='font-size:16px'><thead><tr><td>구분</td><td></td><td></td></tr></thead><tbody><tr><td "
 'rowspan="11">외부요인에 의한 폐질환</td><td>대상이 되는 '
 '질병</td><td>분류번호</td></tr><tr><td>탄광부'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
