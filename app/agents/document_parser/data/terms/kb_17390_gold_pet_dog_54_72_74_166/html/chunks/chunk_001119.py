from langchain_core.documents import Document

chunk = Document(
    page_content=('. 기타 보험회사가 필요하다고 인정하는 서류 및 보험수익자가 보험금의 수령에<br>필요하여 제출하는 서류</p><br><p '
 "id='115' data-category='paragraph' style='font-size:16px'>제5조(보험금의 "
 "분담)</p><br><h1 id='116' style='font-size:16px'>\uf000 회사는 이</h1><br><p "
 "id='117' data-category='paragraph' style='font-size:16px'>특별약관에서 보장하는 위험과 같은 "
 '위험을 보장하는 다른'),
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
