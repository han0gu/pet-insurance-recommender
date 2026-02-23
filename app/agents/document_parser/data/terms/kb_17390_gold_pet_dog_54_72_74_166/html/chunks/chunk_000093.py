from langchain_core.documents import Document

chunk = Document(
    page_content=("지체없이</p><br><p id='114' data-category='paragraph' style='font-size:16px'>회사에 "
 "알려야 합니다.</p><br><p id='115' data-category='paragraph' "
 "style='font-size:16px'>1.</p><br><p id='116' data-category='list' "
 "style='font-size:16px'>보험증권 등에 기재된 직업 또는 직무의 변경<br>가"),
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
