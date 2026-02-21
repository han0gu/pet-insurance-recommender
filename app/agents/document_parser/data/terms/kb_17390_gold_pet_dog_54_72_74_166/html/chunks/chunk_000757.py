from langchain_core.documents import Document

chunk = Document(
    page_content=('사람을 말합니다.</td></tr><tr><td></td><td>계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자에게 보험증권 '
 '드리는 증서를 말합니다.</td></tr><tr><td></td><td>계약을 체결하기 위하여 반려동물이 건강진단을 받아야 하는 계 '
 '진단계약 약을 말합니다.</td></tr><tr><td></td><td>반려동물의 소유와 관련하여 보험사고로 손해를 입은 사람을 말 '
 "피보험자 합니다.</td></tr></tbody></table><p id='111' data-category='paragraph'"),
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
