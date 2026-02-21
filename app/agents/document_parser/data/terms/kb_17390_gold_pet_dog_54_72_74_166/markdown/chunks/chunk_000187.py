from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 보험수익자에게 보험계약대출 사실을 통지할 수 있습니다.\n'
 '# 제36조(중도인출)- \uf000 계약자는 계약일로부터 2년 이상 지난 유효한 계약으로서 계약자의 요청이 있는 경\n'
 '- 우에 한하여 "보험료 및 해약환급금 산출방법서"에 따라 계약자가 요청한 시점에서\n'
 '- 계산된 기본계약 해약환급금과 적립부분 해약환급금 중 적은 금액(적립한 금액에\n'
 '- 서 이 계약에서 정한 대출금이 있을 때에는 그 원금과 이자의 합계액을 차감한 후\n'
 '- 의 잔액을 기준으로 합니다)의 80% 범위 내에서 회사가 정한 방법에 따라 중도인출'),
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
