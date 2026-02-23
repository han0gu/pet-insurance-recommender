from langchain_core.documents import Document

chunk = Document(
    page_content=(". 사망사실을 명확하게 입증할 수 없는 실종, 행방불명 등</p><br><h1 id='70' "
 "style='font-size:16px'>제4조(보험금의 청구)</h1><br><p id='71' "
 "data-category='paragraph' style='font-size:16px'>보험수익자는 다음의 서류를 제출하고 보험금을 "
 "청구하여야 합니다.</p><br><p id='72' data-category='paragraph' "
 "style='font-size:14px'>질</p><p id='73' data-category='list'"),
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
