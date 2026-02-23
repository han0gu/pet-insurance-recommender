from langchain_core.documents import Document

chunk = Document(
    page_content=('- 액을 계약자에게 돌려 드리며, 보험료를 받은 기간에 대하여 평균공시이율 + 해\n'
 '- 1%를 연단위 복리로 계산한 금액을 더하여 지급합니다. 다만, 회사는 계약자가\n'
 '- 제1회 보험료를 신용카드로 납입한 계약의 승낙을 거절하는 경우에는 신용카드\n'
 '# 의 매출을 취소하며 이자를 더하여 지급하지 않습니다.# 제12조(특별약관의 무효)질\n'
 '계약을 맺을 때에 계약에서 정한 반려동물의 나이에 미달되었거나 초과되었을 경우 병\n'
 '이 특별약관은 무효로 하며 이미 납입한 이 특별약관의 보험료를 돌려 드립니다. 다'),
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
