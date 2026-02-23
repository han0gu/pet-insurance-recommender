from langchain_core.documents import Document

chunk = Document(
    page_content=("지난 때에 계약자 또는 보험수익자에게 도달된 것으로 봅니다.</p><h1 id='93' "
 "style='font-size:14px'>제12조(보험수익자의 지정)</h1><br><p id='94' "
 "data-category='paragraph' style='font-size:14px'>\uf000 보험수익자를 지정하지 않은 때에는 "
 '보험수익자를 제9조(만기환급금의 지급) 제1<br>항의 경우는 계약자로 하고, 사망보험금의 경우는 피보험자의 법정상속인으로 하<br>며, '
 '이외의 보험금은 피보험자로 합니다.<br>\uf000 제1항에 따라 지정된 보험수익자가'),
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
