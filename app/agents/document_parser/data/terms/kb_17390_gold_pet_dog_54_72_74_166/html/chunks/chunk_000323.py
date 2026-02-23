from langchain_core.documents import Document

chunk = Document(
    page_content=("data-category='list' style='font-size:16px'>\uf000 회사는 이 계약과 관련된 개인정보를 이 계약의 "
 '체결, 유지, 보험금 지급 등을 위<br>하여"개인정보 보호법","신용정보의 이용 및 보호에 관한 법률" 등 관계 법령에 정<br>한 '
 '경우를 제외하고 계약자, 피보험자 또는 보험수익자의 동의없이 수집, 이용, 조<br>회 또는 제공하지 않습니다'),
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
