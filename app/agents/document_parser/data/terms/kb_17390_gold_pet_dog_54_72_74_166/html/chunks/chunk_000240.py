from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 해지 전에 발생<br>한 보험금 지급사유에 대하여 회사는 보상하여 드립니다.<br>1. 납입최고(독촉)기간 내에 연체보험료를 '
 '납입하여야 한다는 내용<br>2. 납입최고(독촉)기간이 끝나는 날까지 보험료를 납입하지 않을 경우 납입최고<br>(독촉)기간이 끝나는 '
 '날의 다음날에 계약이 해지된다는 내용<br>3'),
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
