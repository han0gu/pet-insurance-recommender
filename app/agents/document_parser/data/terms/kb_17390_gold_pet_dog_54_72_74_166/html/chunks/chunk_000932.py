from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 보통약관 제1절 일반조항 제9조(만기환급금의 지급) 및 제36조(중도</p><br><p id='111' "
 "data-category='paragraph' style='font-size:16px'>인출)은 제외합니다.</p><br><h1 "
 "id='112' style='font-size:20px'>1.</h1><br><p id='113' "
 "data-category='paragraph' style='font-size:20px'>반려동물의료비Ⅱ(강아지)</p><br><p "
 "id='114' data-category='paragraph'"),
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
