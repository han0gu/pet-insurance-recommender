from langchain_core.documents import Document

chunk = Document(
    page_content=(". 계약관계</h1><br><p id='159' data-category='paragraph' "
 "style='font-size:14px'>관련 용어</p><br><table id='160' "
 "style='font-size:14px'><thead><tr><td>용 어</td><td>정 "
 '의</td></tr></thead><tbody><tr><td>계약자</td><td>회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사 '
 '람을 말합니다.</td></tr><tr><td>보험수익자</td><td>보험금 지급사유가 발생하는 때에 회사에 보험금을'),
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
