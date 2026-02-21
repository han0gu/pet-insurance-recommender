from langchain_core.documents import Document

chunk = Document(
    page_content=('. 이 경우 그 대표자는 각각 다른 계약자 또는 보험수익자를 대리하는 것으로 합<br>니다.<br>\uf000 지정된 계약자 또는 '
 '보험수익자의 소재가 확실하지 않은 경우에는 이 계약에 관하<br>여 회사가 계약자 또는 보험수익자 1명에 대하여 한 행위는 각각 다른 '
 "계약자 또는<br>보험수익자에게도 효력이 미칩니다.</p><br><p id='106' "
 "data-category='list'></p><br><table id='107' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>\uf000 계약자가"),
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
