from langchain_core.documents import Document

chunk = Document(
    page_content=('. 피보험자의 임신, 출산(제왕절개를 포함합니다), 산후기. 그러나, 회사가 보<br>장하는 보험금 지급사유와 보장개시일부터 2년이 지난 '
 '후에 발생한 습관성 유<br>산, 불임 및 인공수정 관련 합병증으로 인한 경우에는 보험금을 지급합니다.<br>5'),
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
