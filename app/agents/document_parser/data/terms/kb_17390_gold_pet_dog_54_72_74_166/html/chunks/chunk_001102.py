from langchain_core.documents import Document

chunk = Document(
    page_content=('. 반려동물장례비용지원금(실손)(30일면책)(강아지)【갱신계약】<br>(【갱신계약】은 자동갱신으로 운영합니다)</p><br><p '
 "id='92' data-category='paragraph' style='font-size:14px'>제1조(보험금의 "
 "지급사유)</p><br><h1 id='93' style='font-size:14px'>\uf000 회사는 보험증권에</h1><br><p "
 "id='94' data-category='paragraph' style='font-size:14px'>기재된 반려동물이 이 특별약관의 "
 '보험기간 중'),
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
