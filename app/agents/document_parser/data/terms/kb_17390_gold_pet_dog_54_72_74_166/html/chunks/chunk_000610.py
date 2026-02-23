from langchain_core.documents import Document

chunk = Document(
    page_content=(". 기타 보험수익자가 보험금의 수령에 필요하여 제출하는 서류 질<br>병</p><br><p id='119' "
 "data-category='paragraph' style='font-size:14px'>질</p><br><p id='120' "
 "data-category='paragraph' style='font-size:14px'>병</p><table id='121' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>용 어 풀 이 「건강보험 "
 '행위</td><td>건강보험심사평가원 진료수가코드(EDI)'),
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
