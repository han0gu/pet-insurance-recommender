from langchain_core.documents import Document

chunk = Document(
    page_content=(". 피보험자의 가족관계등록부상 또는 주민등록상의 배우자<br>2. 피보험자의 3촌 이내의 친족</p><br><p id='127' "
 "data-category='paragraph' style='font-size:16px'>\uf000 제1항에도 불구하고, 지정대리청구인이 "
 "지정된 이후에 제38조(적용대상)의 보험수</p><br><p id='128' data-category='paragraph' "
 "style='font-size:14px'>익자가 변경되는 경우에는 이미 지정된 지정대리청구인의 자격은 자동적으로 상실<br>된 것으로 "
 '봅니다'),
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
