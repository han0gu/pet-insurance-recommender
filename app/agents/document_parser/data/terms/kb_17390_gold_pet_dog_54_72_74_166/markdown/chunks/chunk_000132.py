from langchain_core.documents import Document

chunk = Document(
    page_content=('| 적립액(해약환급금)도 | 줄어듭니다.) |\n'
 '| 용 어 풀 이 계약자적립액 장래의 보험금, 해약환급금 등을 지급하기 위하여 계약자가 납입한 보험료 중 | 용 어 풀 이 계약자적립액 '
 '장래의 보험금, 해약환급금 등을 지급하기 위하여 계약자가 납입한 보험료 중 |\n'
 '일정액을 회사가 적립해 둔 금액을 말합니다.- 제23조(보험나이 등)\n'
 '- \uf000 이 약관에서의 피보험자의 나이는 보험나이를 기준으로 합니다. 다만, 제21조(계\n'
 '- 약의 무효) 제2호의 경우에는 실제 만 나이를 적용합니다.'),
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
