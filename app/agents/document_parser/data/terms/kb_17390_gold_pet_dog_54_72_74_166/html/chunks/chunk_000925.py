from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 보험계약이 연<br>장된 경우 연장된 날 기준으로 매년 현재의 예정기초율(적용이율, 적용위험률,<br>부가보험요율) 적용 및 '
 '반려동물의 연령 증가 등의 사유로 보험요율이 변동될 수<br>있으며 이 때의 보험료는 "보험료 및 해약환급금 산출방법서"에 따라 '
 '산출합니<br>다.<br>\uf000 제5항에 따라 보험계약이 연장된 경우 계약자는 그 최초연장된 날로부터 90일 이<br>내에 그 '
 '계약을 취소할 수 있으며, 계약자가 연장된 보험계약을 취소하는 경우 회<br>사는 최초연장된 날 이후 계약자가 납입한 보험료 전액을'),
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
