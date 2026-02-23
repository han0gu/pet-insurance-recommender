from langchain_core.documents import Document

chunk = Document(
    page_content=('치료개시일(해당 상병의 진료를 위하여 최초로 내원(입원을 포함합니<br>다)한 날을 말합니다)로 합니다.<br>\uf000 제2항에서 '
 '"연간"이란 계약일로부터 매1년 단위로 도래하는 계약해당일 전까지<br>기간을 의미합니다.</p><br><p id=\'107\' '
 "data-category='paragraph' style='font-size:16px'>제2조(보험금 지급에 관한 "
 "세부규정)</p><br><p id='108' data-category='paragraph' "
 "style='font-size:16px'>보험수익자와 회사가"),
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
