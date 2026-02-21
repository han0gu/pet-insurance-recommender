from langchain_core.documents import Document

chunk = Document(
    page_content=("부활이 이루어진 경우 부활을 청약한 날을 제5항의 청약일로 하여 적용합니다.</p><br><table id='177' "
 "style='font-size:14px'><thead><tr><td>용 어 "
 '풀</td><td>이</td></tr></thead><tbody><tr><td colspan="2">∙ 보험가입금액 제한 피보험자가 '
 '가입을 할 수 있는 최대 보험가입금액을 제한하는 방법을 말합니다'),
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
