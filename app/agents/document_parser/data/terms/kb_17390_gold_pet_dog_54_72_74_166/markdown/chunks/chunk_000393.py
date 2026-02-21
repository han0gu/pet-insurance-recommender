from langchain_core.documents import Document

chunk = Document(
    page_content=('- 해당하는 경우에 한하여 해당 보험금을 지급합니다. 다만, 제4조(특별약관의 소\n'
 '- 멸) 제2항에 따라 이 특별약관의 계약자적립액 등을 지급한 경우에는, 이미 지급\n'
 '- 된 계약자적립액 등을 차감하고 그 차액을 지급합니다.\n'
 '# 제3조(6대호흡계특정질환의 정의 및 진단확정)\uf000 이 특별약관에 있어서 "6대호흡계특정질환"이라 함은 제9차 '
 '한국표준질병․사인분\n'
 '특\n'
 '류에 있어서 【별표12】(6대호흡계특정질환 분류표)에서 정한 질병을 말합니다.\n'
 '별\n'
 '\uf000 "6대호흡계특정질환"의 진단확정은 의료법 제3조(의료기관)에서 정한 국내의 병\n'
 '약'),
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
