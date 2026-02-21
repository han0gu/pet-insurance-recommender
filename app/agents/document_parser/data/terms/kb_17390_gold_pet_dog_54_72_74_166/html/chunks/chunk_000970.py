from langchain_core.documents import Document

chunk = Document(
    page_content=('코로나바이러스감염 증, 렙토스피라감염증, 필라리아감염증, 광견병, 인플루엔자 '
 "감염,</td></tr></tbody></table><br><p id='162' data-category='list' "
 "style='font-size:16px'>켄넬코프<br>3"),
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
