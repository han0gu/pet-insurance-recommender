from langchain_core.documents import Document

chunk = Document(
    page_content=('. (금융감독원 홈페이지(www.fss.or.kr) 의 "업무자료-보험상품자료"에서 확인할 수 '
 '있습니다.)</td></tr><tr><td>해약환급금</td><td>계약이 해지되는 때에 회사가 계약자에게 돌려주는 금액을 '
 '말합니다.</td></tr><tr><td>이미 납입한</td><td>보험료 계약자가 실제 납입한 보험료를 '
 "말합니다.</td></tr></tbody></table><p id='8' data-category='paragraph' "
 "style='font-size:16px'>4"),
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
