from langchain_core.documents import Document

chunk = Document(
    page_content=(". 현재의 직업을 그만둔 경우</p><p id='117' data-category='paragraph' "
 "style='font-size:16px'>부 가 설 명 직업 또는 직무</p><br><p id='118' "
 "data-category='paragraph' style='font-size:16px'>∙ 직업</p><br><p id='119' "
 "data-category='paragraph' style='font-size:16px'>1) 생계유지 등을 위하여 일정한 기간동안(예: "
 "6개월 이상) 계속하여 종사</p><br><p id='120'"),
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
