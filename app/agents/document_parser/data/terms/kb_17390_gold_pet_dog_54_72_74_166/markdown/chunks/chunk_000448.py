from langchain_core.documents import Document

chunk = Document(
    page_content=('| 자기부담금 | 보험사고로 인하여 발생한 손해에 대하여 계약자 또는 피보 험자가 부담하는 일정 금액을 말합니다. |\n'
 '- 100 -| 용 |  |\n'
 '| --- | --- |\n'
 '| 보험금 | 어 정 의 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 분담 계약(공제계약을 포함합니다)이 있을 경우 비율에 '
 '따라 손 |\n'
 '| 해를 | 보상합니다. |\n'
 '| --- | --- |'),
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
