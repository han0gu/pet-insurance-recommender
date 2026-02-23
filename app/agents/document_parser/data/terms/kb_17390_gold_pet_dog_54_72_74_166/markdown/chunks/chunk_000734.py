from langchain_core.documents import Document

chunk = Document(
    page_content=('반예 시반려동물 위탁비용이![image](/image/placeholder)\n'
 '- Chart Type: bar\n'
 '|  | 보호 | 분쟁세외 | 부장 |\n'
 '| --- | --- | --- | --- |\n'
 '| item_01 | 180Not specified | 180Not specified | 180Not specified |\n'
 '려동\uf000- 127 -KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 127- 끝났을 때에도 퇴원하기 전까지의 계속중인 '
 '입원기간에 대하여는 제1조(보험금의'),
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
