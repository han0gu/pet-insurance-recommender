from langchain_core.documents import Document

chunk = Document(
    page_content=('- 를 집행하는 그 밖의 기관)을 말합니다.\n'
 '- 1) 기명피보험자(가입동물의 소유자에 한함) 및 기명피보험자의 배우자\n'
 '- 2) 기명피보험자나 배우자와 생계를 함께하는 동거 친족 및 별거하는 미혼자녀\n'
 '- 다. 보험증권: 계약의 성립과 그 내용을 증명하기 위하여 회사가 계약자에게 드리는 증서를 말\n'
 '- 합니다.\n'
 "- 라. 갱신: 동일 보험상품('반려견보험 애니펫'), 유사보험상품(파밀리아리스 애견의료보험2) 또는\n"
 '- 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 계약 중 회사가 유사하다고 판단'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
