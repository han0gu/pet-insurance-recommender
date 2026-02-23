from langchain_core.documents import Document

chunk = Document(
    page_content=("경우에는 제6항 내지 제7항을 적용하지 않습니다.</p><br><h1 id='224' "
 "style='font-size:16px'>제2조(보험금 지급에 관한 세부규정)</h1><br><h1 id='225' "
 "style='font-size:16px'>\uf000 보험증권에 기재된 반려동물이</h1><br><p id='226' "
 "data-category='paragraph' style='font-size:16px'>동일한 날에 두 종류 이상의 "
 '"반려동물주요치료"</p><br><p id=\'227\' data-category=\'paragraph\''),
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
