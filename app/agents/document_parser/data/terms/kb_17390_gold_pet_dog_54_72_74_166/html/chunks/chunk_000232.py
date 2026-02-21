from langchain_core.documents import Document

chunk = Document(
    page_content=('납입최고(독촉)기간이 지나기 전까지 회사가 정한 방법에 따라<br>보험료의 자동대출납입을 신청할 수 있으며, 이 경우 '
 '제35조(보험계약대출) 제1항<br>에 따른 보험계약대출금으로 보험료가 자동으로 납입되어 계약은 유효하게 지속됩<br>니다'),
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
