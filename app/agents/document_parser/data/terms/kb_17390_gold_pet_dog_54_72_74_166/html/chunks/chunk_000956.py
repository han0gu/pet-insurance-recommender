from langchain_core.documents import Document

chunk = Document(
    page_content=('\uf000 제1항에서 \'연간\'이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전일까지" '
 'data-coord="top-left:(152,833); bottom-right:(738,938)" '
 "/></figure></td></tr></tbody></table><br><p id='146' "
 "data-category='paragraph' style='font-size:14px'>기간을 의미합니다.</p><br><p "
 "id='147' data-category='paragraph' style='font-size:14px'>110 KB 금쪽같은"),
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
