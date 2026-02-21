from langchain_core.documents import Document

chunk = Document(
    page_content=('. 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류<br>제1항 제2호의 사고증명서는 의료법 제3조(의료기관)에서 규정한 국내의 '
 "병원</p><br><h1 id='82' style='font-size:14px'>\uf000</h1><br><p id='83' "
 "data-category='paragraph' style='font-size:14px'>이나 의원 또는 국외의 의료관련법에서 정한 "
 "의료기관에서 발급한 것이어야 합<br>니다.</p><br><p id='84' data-category='paragraph'"),
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
