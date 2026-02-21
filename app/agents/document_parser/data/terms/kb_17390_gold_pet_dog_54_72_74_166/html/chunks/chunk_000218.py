from langchain_core.documents import Document

chunk = Document(
    page_content=('. 예) 1년치 보험료(연납)을 받은 후 6개월이 경과했다면, 6개월(미경과 기간)에 대응하는 것으로 미경과보험료라고 '
 "합니다.</td></tr></tbody></table><br><p id='27' data-category='paragraph' "
 "style='font-size:14px'>법</p><p id='28' data-category='paragraph' "
 "style='font-size:16px'>KB 금쪽같은 펫보험(강아지)(무배당)(26.01) 65</p><br><p id='29' "
 "data-category='paragraph'"),
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
