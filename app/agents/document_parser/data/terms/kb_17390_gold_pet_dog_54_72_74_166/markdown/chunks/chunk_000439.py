from langchain_core.documents import Document

chunk = Document(
    page_content=('- (Fiberglass Cast)를 병변이 있는 뼈, 관절부위의 둘레 모두에 착용시켜\n'
 '- (Circular Cast) 감은 다음 굳어지게 하여 치료 효과를 가져 오는 치료법을 말합\n'
 '- 니다. 단, 부목(Splint Cast)치료는 제외합니다.\n'
 '- \uf000 제1항의 "부목(Splint Cast)치료"라 함은 석고붕대 또는 섬유유리붕대(Fiberglass\n'
 '- Cast)를 고정할 부분의 일측면 또는 양측면에 착용시키고 대주는 치료법을 말합니다.'),
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
