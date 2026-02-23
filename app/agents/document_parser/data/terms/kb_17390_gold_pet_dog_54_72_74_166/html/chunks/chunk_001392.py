from langchain_core.documents import Document

chunk = Document(
    page_content=("보장개시)<br>제2조(보장특약의 자동갱신)의 규정에 따라 계약이 갱신되는 경우, 갱신보장특약의</p><br><p id='38' "
 "data-category='paragraph' style='font-size:14px'>보장개시는 갱신일 당일로 합니다.</p><p "
 "id='39' data-category='paragraph' style='font-size:14px'>제6조(준용규정)</p><br><p "
 "id='40' data-category='paragraph' style='font-size:14px'>이 특약에서 정하지 않은 사항은"),
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
