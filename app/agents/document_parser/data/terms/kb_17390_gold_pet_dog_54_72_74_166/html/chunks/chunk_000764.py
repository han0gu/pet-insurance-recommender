from langchain_core.documents import Document

chunk = Document(
    page_content=("id='116' data-category='paragraph' style='font-size:16px'>- 100 -</p><table "
 "id='117' "
 "style='font-size:16px'><thead><tr><td>용</td><td></td></tr></thead><tbody><tr><td>보험금</td><td>어 "
 '정 의 이 계약에서 보장하는 위험과 같은 위험을 보장하는 다른 분담 계약(공제계약을 포함합니다)이 있을 경우 비율에 따라 '
 "손</td></tr></tbody></table><br><table id='118'"),
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
