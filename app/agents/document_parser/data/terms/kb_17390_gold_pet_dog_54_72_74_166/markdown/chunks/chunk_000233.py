from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1조(보험금의 지급사유)의 "사망"에는 보험기간에 다음 어느 하나의 사유가 발- 생한 경우를 포함합니다.\n'
 '- 1. 실종선고를 받은 경우: 법원에서 인정한 실종기간이 끝나는 때에 사망한 것으\n'
 '- 로 봅니다.\n'
 '- 2. 관공서에서 수해, 화재나 그 밖의 재난을 조사하고 사망한 것으로 통보하는\n'
 '- 경우: 가족관계등록부에 기재된 사망연월일을 기준으로 합니다.'),
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
