from langchain_core.documents import Document

chunk = Document(
    page_content=('# 반한 계약을# 말합니다.∙ 제척기간\n'
 '어떤 종류의 권리에 대하여 법률이 정하고 있는 존속 기간을 말하며, 이 기간이# 지나면 권리가 소멸됩니다.제32조(중대사유로 인한 '
 '해지)\uf000 회사는 아래와 같은 사실이 있을 경우에는 안 날부터 1개월 이내에 계약을 해지할\n'
 '수 있습니다.1. 계약자, 피보험자 또는 보험수익자가 보험금을 지급받을 목적으로 고의로 보험\n'
 '금 지급사유를 발생시킨 경우\n'
 '2. 계약자, 피보험자 또는 보험수익자가 보험금 청구에 관한 서류에 고의로 사실과'),
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
