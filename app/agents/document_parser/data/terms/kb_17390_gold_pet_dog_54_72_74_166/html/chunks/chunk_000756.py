from langchain_core.documents import Document

chunk = Document(
    page_content=(". 계약관계 관련 용어</h1><br><table id='110' "
 "style='font-size:14px'><thead><tr><td>용</td><td>어 정 "
 '의</td></tr></thead><tbody><tr><td></td><td>회사와 계약을 체결하고 보험료를 납입할 의무를 지는 사람을 '
 '계약자 말합니다.</td></tr><tr><td></td><td>보험금 지급사유가 발생하는 때에 회사에 보험금을 청구하여 받 보험수익자 '
 '을 수 있는 사람을 말합니다.</td></tr><tr><td></td><td>계약의 성립과 그 내용을'),
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
