from langchain_core.documents import Document

chunk = Document(
    page_content=('. 질</p><h1 id=\'230\' style=\'font-size:16px\'>제3조("반려동물주요치료"의 '
 "정의)</h1><br><p id='231' data-category='paragraph' "
 'style=\'font-size:16px\'>\uf000 이 특별약관에 있어서 "반려동물주요치료"라 함은 국내에서 수의사가 '
 "보험증권에<br>기재된 반려동물에게 시행한 치료로서 다음 각 호의 사항을 말합니다.</p><br><p id='232' "
 "data-category='list' style='font-size:16px'>1"),
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
