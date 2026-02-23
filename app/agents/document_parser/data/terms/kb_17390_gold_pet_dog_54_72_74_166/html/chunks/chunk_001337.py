from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 이 특별약관의 보험금 지급<br>일 이후 사망보장특별약관에 정한 사망보험금의 청구를 받아도 이 특별약관에 의<br>하여 '
 '지급된 보험금액에 해당하는 사망보험금은 지급하지 않습니다.<br>\uf000 이 특별약관의 보험금이 지급되기 전에 사망보장특별약관에 정한 '
 '사망보험금의 청<br>구를 받았을 경우 이 특별약관의 보험금 청구가 있어도 이를 없었던 것으로 보아<br>이 특별약관의 보험금을 지급하지 '
 '않습니다.<br>\uf000 사망보장특별약관에 정한 사망보험금이 지급된 때에는 그 이후 이 특별약관의 보<br>험금을 지급하지 '
 '않습니다.<br>\uf000'),
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
