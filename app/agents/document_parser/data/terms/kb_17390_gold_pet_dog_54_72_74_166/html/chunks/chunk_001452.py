from langchain_core.documents import Document

chunk = Document(
    page_content=('. 또한 유사계약이 다수인<br>경우 해당 반려동물에게 가장 유리한 계약조건을 적용합니다.<br>단, 유사계약 청약일 이후 제1항에서 '
 '정한 질병과 관련한 새로운 위험(재진단·치<br>료 등은 해당하지 않습니다)이 발생하거나, 새로운 질병에 대한 보장이 '
 '추가(입원<br>비, 수술비, 진단비 등 보장 범위의 변경 또는 확대는 해당하지 않습니다)된 경우<br>이를 적용하지 않을 수 '
 '있습니다.<br>\uf000 제1항에도 불구하고 다음 사항 중 어느 한가지의 경우에 해당되는 사유로 보험계<br>약에서 정한 보험금의 '
 '지급사유가 발생한 경우에는'),
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
