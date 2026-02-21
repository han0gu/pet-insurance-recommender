from langchain_core.documents import Document

chunk = Document(
    page_content=('(복합레진, 인레이, 온레이 등)한 치아, 기존 의치(틀니, 임플란트 등)\n'
 '의 결손은 치아의 상실로 인정하지 않는다.\n'
 '14) 상실된 치아의 크기가 크든지 또는 치간의 간격이나 치아 배열구조 등의\n'
 '문제로 사고와 관계없이 새로운 치아가 결손된 경우에는 사고로 결손된\n'
 '치아 수에 따라 지급률을 결정한다.\n'
 '15) 어린이의 유치는 향후에 영구치로 대체되므로 후유장해의 대상이 되지\n'
 '않으나, 선천적으로 영구치 결손이 있는 경우에는 유치의 결손을 후유\n'
 '장해로 평가한다.'),
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
