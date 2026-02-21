from langchain_core.documents import Document

chunk = Document(
    page_content=('1. 제4조(보상하는 손해의 범위) 제1호의 손해배상금 : 보상한도액을 한도로 보- 상하되, 자기부담금이 약정된 경우에는 그 자기부담금을 '
 '초과한 부분만 보상\n'
 '- 합니다.\n'
 '- 2. 제4조(보상하는 손해의 범위) 제2호의 "가"목, "나"목 또는 "마"목의 비용 :\n'
 '122 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)비용의 전액을 보상합니다.3. 제4조(보상하는 손해의 범위) 제2호의 "다"목 '
 '또는 "라"목의 비용 : 이 비용# 과 제1호에 의한 보상액의합계액을 보상한도액내에서 보상합니다.- 제10조(의무보험과의 관계)'),
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
