from langchain_core.documents import Document

chunk = Document(
    page_content=('. 제4조(보상하는 손해의 범위) 제2호의 "다"목 또는 "라"목의 비용 : 이 비용</p><br><h1 id=\'184\' '
 "style='font-size:14px'>과 제1호에 의한 보상액의</h1><br><p id='185' "
 "data-category='paragraph' style='font-size:14px'>합계액을 보상한도액내에서 "
 "보상합니다.</p><br><p id='186' data-category='list' "
 "style='font-size:14px'>제10조(의무보험과의 관계)<br>\uf000 회사는 이 특별약관에 따라 보상하여야"),
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
