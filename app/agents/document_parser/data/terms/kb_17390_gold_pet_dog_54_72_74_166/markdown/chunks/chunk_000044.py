from langchain_core.documents import Document

chunk = Document(
    page_content=('을 지급합니다.- \n'
 '58 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)|  | '
 '<table><thead></thead><tbody><tr><td></td></tr><tr><td>예 시 보험금을 나누어 지급받을 '
 '경우</td></tr></tbody></table> |\n'
 '| --- | --- |'),
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
