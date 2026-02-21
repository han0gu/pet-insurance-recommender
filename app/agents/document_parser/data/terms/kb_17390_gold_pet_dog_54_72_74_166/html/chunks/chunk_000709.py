from langchain_core.documents import Document

chunk = Document(
    page_content=('보험수익자에게 지급합니다.<br>\uf000 제1항에서 "연간"이란 계약일부터 매1년 단위로 도래하는 계약해당일 전일까지<br>기간을 '
 "의미합니다.</p><br><h1 id='13' style='font-size:14px'>제1조(보험금의</h1><p id='14' "
 "data-category='paragraph' style='font-size:14px'>제2조(보험금 지급에 관한 "
 "세부규정)</p><br><h1 id='15' style='font-size:14px'>\uf000 보험수익자와 회사가</h1><br><p "
 "id='16'"),
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
