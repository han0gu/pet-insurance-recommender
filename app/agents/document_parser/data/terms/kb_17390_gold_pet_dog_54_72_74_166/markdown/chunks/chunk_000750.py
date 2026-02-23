from langchain_core.documents import Document

chunk = Document(
    page_content=('- 관도 더 이상 효력을 가지지 않습니다.\n'
 '- \uf000 이 특별약관은 피보험자가 이륜자동차를 소유, 사용(직업, 직무 또는 동호회 활동\n'
 '- 과 출퇴근용도 등으로 주로 사용하는 경우에 한하며 일회적인 사용은 제외), 관리\n'
 '- 하는 경우에 한하여 부가하여 이루어집니다.\n'
 '제2조(보험금을 지급하지 않는사유)- \uf000 회사는 보험계약의 내용에도 불구하고 보험증권에 기재된 보험기간 중에 이륜자동\n'
 '- 차를 운전(탑승을 포함합니다. 이하 같습니다)하던 중 발생한 급격하고도 우연한'),
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
