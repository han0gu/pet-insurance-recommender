from langchain_core.documents import Document

chunk = Document(
    page_content=('. 물<br>\uf000 제1항에 따라 제출한 장애인증명서의 장애기간이 변경되는 경우 계약자는 이를 회<br>사에 알리고 변경된 '
 "장애기간이 기재된 장애인증명서를 제출하여야 합니다.<br>제</p><br><p id='90' "
 "data-category='list'></p><br><h1 id='91' "
 "style='font-size:16px'>제3조(장애인전용보험으로의 전환)</h1><br><h1 id='92' "
 "style='font-size:16px'>\uf000 회사는 이 특별약관이 부가된</h1><br><p id='93'"),
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
