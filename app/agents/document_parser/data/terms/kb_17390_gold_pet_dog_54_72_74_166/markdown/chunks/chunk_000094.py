from langchain_core.documents import Document

chunk = Document(
    page_content=('- 위 복리로 계산한 금액을 더하여 지급합니다. 다만, 회사는 계약자가 제1회 보험료\n'
 '- 를 신용카드로 납입한 계약의 승낙을 거절하는 경우에는 신용카드의 매출을 취소하\n'
 '- 며 이자를 더하여 지급하지 않습니다.\n'
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 61- 61 -\uf000 회사가 제2항에 따라 일부보장 제외 조건을 붙여 '
 '승낙하였더라도 청약일로부터 5년\n'
 '(갱신형 계약의 경우에는 최초 계약의 청약일로부터 5년)이 지나는 동안 보장이 제\n'
 '외되는 질병으로 추가 진단(단순 건강검진 제외) 또는 치료 사실이 없을 경우, 청'),
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
