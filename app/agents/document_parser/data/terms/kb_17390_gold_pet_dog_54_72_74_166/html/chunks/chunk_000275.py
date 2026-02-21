from langchain_core.documents import Document

chunk = Document(
    page_content=("id='94' data-category='paragraph' style='font-size:14px'>제33조(회사의 파산선고와 "
 "해지)</p><br><p id='95' data-category='paragraph' "
 "style='font-size:14px'>\uf000 회사가 파산의 선고를 받은 때에는</p><br><p id='96' "
 "data-category='paragraph' style='font-size:14px'>계약자는 계약을 해지할 수 "
 "있습니다.</p><br><p id='97' data-category='paragraph'"),
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
