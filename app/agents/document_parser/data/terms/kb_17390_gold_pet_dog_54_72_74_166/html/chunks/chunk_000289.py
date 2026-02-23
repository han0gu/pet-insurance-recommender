from langchain_core.documents import Document

chunk = Document(
    page_content=('적립부분 해약환급금 중 적은 금액이 100만원인 경우 통 ⇒ 총 중도인출 가능액 = 100만원× 80% = 80만원 사항 ⇒ 기 신청한 '
 "대출금이 있는 경우(원금과 이자의 합계를 10만원으로 가정)</td></tr></tbody></table><br><p id='115' "
 "data-category='paragraph' style='font-size:16px'>중도인출 가능액 = 80만원(총 중도인출 가능액) "
 "- 10만원 = 70만원</p><p id='116' data-category='paragraph'"),
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
