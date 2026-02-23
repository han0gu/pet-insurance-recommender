from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:14px'>\uf000 보험증권에 기재된 반려동물이 보험기간 중에 사망하여 보험의 목적에 대해 "
 '이<br>질<br>특별약관에서 정한 보험금 지급사유가 더 이상 발생할 수 없는 경우 회사는 "보 병<br>험료 및 해약환급금 '
 '산출방법서"에서 정하는 바에 따라 반려동물 사망 당시 이<br>특별약관의 계약자적립액 및 미경과보험료를 계약자에게 '
 '지급합니다.<br>\uf000 보험의 목적이 다수인 경우 제1항은 보험의 목적별로 각각 적용합니다.<br>반</p><br><p '
 "id='177' data-category='paragraph'"),
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
