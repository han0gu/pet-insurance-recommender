from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 회사는 납입최고(독촉)기간 안에 발생한 사고에 대하여 약정한 보험금을 지급합니 약\n'
 'KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 135- 135 -다. 이 경우 계약자는 즉시 갱신보장특약의 보험료를 납입하여야 '
 '합니다. 만약, 이보험료를 납입하지않으면 회사는 지급할 보험금에서 이를 공제할 수 있습니다.제5조(갱신보장특약의 보장개시)'),
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
