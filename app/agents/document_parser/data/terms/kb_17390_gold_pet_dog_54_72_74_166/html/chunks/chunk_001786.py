from langchain_core.documents import Document

chunk = Document(
    page_content=('. 진단서 상의 분류번호는 한국표준질병․사인분류 질병코딩지침서에 따라 기재<br>된 것을 인정합니다.<br>\uf000 진단 당시의 '
 "한국표준질병․사인분류에 따라 이 약관에서 보장하는 질병에 대한 보</p><p id='105' "
 "data-category='paragraph' style='font-size:18px'>- 162 -</p><br><p id='106' "
 "data-category='paragraph' style='font-size:14px'>험금 지급여부가 판단된 경우, 이후 "
 '한국표준질병․사인분류 개정으로 질병 분류가<br>변경되더라도'),
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
