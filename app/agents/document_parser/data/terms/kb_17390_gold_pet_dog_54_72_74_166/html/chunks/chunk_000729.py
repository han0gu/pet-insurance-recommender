from langchain_core.documents import Document

chunk = Document(
    page_content=(". 질병을 원인으로 하지 않은 신체검사, 예방접종, 인공유산, 불임시술, 제왕절</p><br><p id='45' "
 "data-category='list' style='font-size:16px'>개수술<br>2. 피로, 권태, 심신허약 등을 치료하기 "
 "위한 안정치료비</p><br><p id='46' data-category='paragraph' "
 "style='font-size:16px'>않</p><br><p id='47' data-category='list'></p><br><h1 "
 "id='48' style='font-size:16px'>3"),
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
