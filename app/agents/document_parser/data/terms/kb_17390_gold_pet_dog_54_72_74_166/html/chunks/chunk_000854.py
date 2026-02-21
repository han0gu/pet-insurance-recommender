from langchain_core.documents import Document

chunk = Document(
    page_content=('있습니다.<br>\uf000 제1항에 따라 계약이 취소된 경우에는 회사는 이미 납입한 보험료를 '
 "계약자에게<br>돌려드립니다.</p><br><p id='23' data-category='list' "
 "style='font-size:14px'>제11조(보험계약의 성립)<br>\uf000 계약은 계약자의 청약과 회사의 승낙으로 "
 '이루어집니다.<br>특<br>\uf000 회사는 반려동물이 이 특별약관에 적합하지 않은 경우에는 승낙을 거절하거나<br>별<br>별도의 '
 '조건(보험가입금액 제한, 일부보장 제외, 보험금 삭감, 보험료 할증<br>약<br>등)을 붙여 승낙할 수'),
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
