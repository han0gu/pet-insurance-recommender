from langchain_core.documents import Document

chunk = Document(
    page_content=('보험료 할증과 같이 조건부로 승낙하 는 등 계약 승낙에 영향을 미칠 수 있는 사항을 '
 '말합니다.</td></tr><tr><td>보험가입금액</td><td>회사와 계약자간에 약정한 금액으로 보험사고가 발생할 때 회사가 지급할 '
 '최대 보험금을 말합니다.</td></tr><tr><td>자기부담금</td><td>보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보 '
 "험자가 부담하는 일정 금액을 말합니다.</td></tr></tbody></table><p id='116' "
 "data-category='paragraph'"),
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
