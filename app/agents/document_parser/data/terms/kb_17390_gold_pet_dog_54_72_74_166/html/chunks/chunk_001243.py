from langchain_core.documents import Document

chunk = Document(
    page_content=("특별약관 - 보장특약 자동갱신(추가납입</p><br><p id='55' data-category='list' "
 'style=\'font-size:14px\'>형) 특별약관"에 의해 계약자의 선택에 따라 자동갱신으로 운영합니다'),
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
