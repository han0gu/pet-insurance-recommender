from langchain_core.documents import Document

chunk = Document(
    page_content=('과목마다 전속하는 전문의를 둘 것제15조(상해보험계약 후 알릴 의무) 통약\n'
 '\uf000 계약자 또는 피보험자는 보험기간 중에 피보험자에게 다음 각 호의 변경이 발생한경우에는 우편, 전화, 방문 등의 방법으로 '
 '지체없이회사에 알려야 합니다.1.- 보험증권 등에 기재된 직업 또는 직무의 변경\n'
 '- 가. 현재의 직업 또는 직무가 변경된 경우\n'
 '- 나. 직업이 없는 자가 취직한 경우\n'
 '- 다. 현재의 직업을 그만둔 경우'),
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
