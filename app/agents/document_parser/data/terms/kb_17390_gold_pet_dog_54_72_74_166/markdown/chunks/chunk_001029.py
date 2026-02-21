from langchain_core.documents import Document

chunk = Document(
    page_content=('| 외상후스트레스장애 F43.1 주) 1. 대상질병 분류표의 분류번호와 다르나 한국표준질병․사인분류의 기준에 따라 분류번호를 동시에 '
 '부여가 가능한 경우 대상질병 분류에 포함합니다. 2. 제10차 개정 이후 이 약관에서 보장하는 특정정신질환 해당여부는 피보험자가 진단된 '
 '당시 시행되고 있는 한국표준질병․사인분류에 따라 판단합니다. 3. 진단서 상의 분류번호는 한국표준질병․사인분류 질병코딩지침서에 따라 기재 '
 '된 것을 인정합니다'),
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
