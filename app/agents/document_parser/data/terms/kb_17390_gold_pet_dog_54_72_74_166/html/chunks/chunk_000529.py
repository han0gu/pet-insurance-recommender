from langchain_core.documents import Document

chunk = Document(
    page_content=("style='font-size:16px'>기간을 의미합니다.</p><p id='275' data-category='paragraph' "
 "style='font-size:16px'>제2조(보험금 지급에 관한 세부규정)<br>보험수익자와 회사가 제1조(보험금의 지급사유)의 "
 '보험금 지급사유에 대해 합의하지<br>못할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있<br>습니다'),
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
