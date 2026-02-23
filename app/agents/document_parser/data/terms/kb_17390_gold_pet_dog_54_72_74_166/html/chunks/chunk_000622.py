from langchain_core.documents import Document

chunk = Document(
    page_content=("id='146' data-category='list' style='font-size:14px'>제2조(보험금 지급에 관한 "
 '세부규정)<br>\uf000 "호스피스·완화의료 및 임종과정에 있는 환자의 연명의료 결정에 관한 법률"에<br>따른 연명의료중단등결정 및 '
 '그 이행으로 피보험자가 사망하는 경우 연명의료중<br>단등결정 및 그 이행은 제1조(보험금의 지급사유) "사망"의 원인 및 '
 '"사망보험금<br>"지급에 영향을 미치지 않습니다.<br>\uf000 보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 '
 '대해 합의<br>하지 못할 때는'),
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
