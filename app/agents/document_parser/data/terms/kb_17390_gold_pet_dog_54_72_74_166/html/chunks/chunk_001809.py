from langchain_core.documents import Document

chunk = Document(
    page_content=("id='3' data-category='paragraph' style='font-size:14px'>164 KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01)</p><br><p id='4' data-category='paragraph' "
 "style='font-size:18px'>- 164 -</p><table id='5' "
 "style='font-size:14px'><thead><tr><td></td><td>코드 "
 '특정</td><td></td></tr></thead><tbody><tr><td rowspan="25"></td><td'),
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
