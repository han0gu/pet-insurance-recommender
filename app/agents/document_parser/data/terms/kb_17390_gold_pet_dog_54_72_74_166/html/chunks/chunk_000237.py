from langchain_core.documents import Document

chunk = Document(
    page_content=('범위 내에서 납입할 보험료를 자동적으로 대출하 여 이를 보험료 납입에 충당하는 서비스를 '
 "말합니다.</td></tr></tbody></table><p id='47' data-category='paragraph' "
 "style='font-size:14px'>\uf000 계약자가 제2회 이후의 보험료를 납입기일까지 납입하지 않아 보험료 "
 "납입이</p><br><p id='48' data-category='paragraph' "
 "style='font-size:14px'>연체</p><br><p id='49' data-category='paragraph'"),
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
