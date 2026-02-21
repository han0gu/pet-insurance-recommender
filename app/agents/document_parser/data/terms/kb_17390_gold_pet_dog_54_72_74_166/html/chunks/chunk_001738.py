from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="27">창상봉합술Ⅱ (급여) (안면/경부) 160</td><td>대상이 되는 '
 '항목</td><td>수가코드</td></tr><tr><td>창상봉합술</td><td></td></tr><tr><td>가'),
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
