from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만,<br>진단계약에서 진단을 받지 않은 경우라도 상해로 보험금 지급사유가 발생하는<br>경우에는 보장을 '
 '해드립니다.<br>\uf000 제1항의 보험료는 제3조(보험금의 지급사유) 에 정한 보험금의 지급에 필요한 보</p><br><p '
 "id='36' data-category='paragraph' style='font-size:14px'>험료(이하 "
 '"보장보험료"라 합니다)와 회사가 적립한 금액을 돌려주는데 필요한 보<br>험료(이하 "적립보험료"라 합니다)로 구성됩니다.(이하 '
 '"보장보험료"와 "적립보<br>험료"를 합하여 "보험료"라'),
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
