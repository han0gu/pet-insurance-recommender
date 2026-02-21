from langchain_core.documents import Document

chunk = Document(
    page_content=("합니다.</p><br><p id='29' data-category='paragraph' "
 "style='font-size:14px'>제2조(보험금 지급에 관한 세부규정)</p><br><p id='30' "
 "data-category='paragraph' style='font-size:14px'>질병입원일당은 같은 질병의 치료를 "
 "목적으로</p><br><p id='31' data-category='paragraph' "
 "style='font-size:14px'>\uf000 제1조(보험금의 지급사유) 제1항의</p><br><p id='32'"),
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
