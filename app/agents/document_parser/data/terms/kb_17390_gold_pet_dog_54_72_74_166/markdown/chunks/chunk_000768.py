from langchain_core.documents import Document

chunk = Document(
    page_content=('- 또한, 회사가 지정한 의사에 의한 피보험자의 진단을 요구한 경우에는 진단을 받\n'
 '- 지 않는 때에는 진단을 받고 사실 확인이 끝날 때까지 이 특별약관의 보험금을 지\n'
 '- 급하지 않습니다. 반\n'
 '- \uf000 회사는 제1항 및 제2항의 규정에 의한 지급기일 내에 이 특별약관의 보험금을 지 려\n'
 '- 급하지 않았을 때에는 그 지급기일의 다음날부터 지급일까지의 기간에 대하여 "보 동\n'
 '- 험금을 지급할 때의 적립이율 계산(【별표2】참조)"에서 정한 이율로 계산한 금액 물\n'
 '- 을 더하여 지급합니다.'),
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
