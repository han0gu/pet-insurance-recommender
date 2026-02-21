from langchain_core.documents import Document

chunk = Document(
    page_content=('우에 한하여 해당 보험금을 지급합니다. 다만, 제4조(특별약관의 소멸) 제2항에따라 이 특별약관의 계약자적립액 등을 지급한 경우에는, '
 '이미 지급된 계약자적# 립액 등을 차감하고 그 차액을지급합니다.- 제3조(천식지속상태의 정의 및 진단 확정)\n'
 '- \uf000 이 특별약관에 있어서 "천식지속상태"라 함은 제9차 한국표준질병․사인분류에 있\n'
 '- 어서 【별표13】(천식지속상태 분류표)에서 정한 질병을 말합니다.\n'
 '- \uf000 "천식지속상태"의 진단확정은 의료법 제3조(의료기관)에서 정한 국내의 병원이나'),
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
