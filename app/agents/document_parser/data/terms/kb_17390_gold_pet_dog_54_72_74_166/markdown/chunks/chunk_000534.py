from langchain_core.documents import Document

chunk = Document(
    page_content=('- \uf000 이 계약의 보험기간 종료 후 계약자가 재가입을 원하는 경우 계약자는 재가입 시\n'
 '점에서 회사가 판매하는 동일하거나 객관적이고 합리적인 범위내에서 기존 계약108 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)# '
 '내용에 상응한 반려동물보험 상품(보험업감독규정 제1-2조(정의)에서 정한 장기\n'
 '손해보험에 한하며 이하 "반려동물보험 상품"이라 합니다)으로 가입을 할 수 있'),
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
