from langchain_core.documents import Document

chunk = Document(
    page_content=('공시이율 규정<br>유배당 보험계약의 경우 계약자 배당에 관한 사항<br>그 밖에 약관에 기재된 보험계약의 중요사항</p><br><p '
 "id='232' data-category='paragraph' style='font-size:16px'>∙</p><p id='233' "
 "data-category='paragraph' style='font-size:16px'>KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01) 63</p><br><p id='234' data-category='paragraph'"),
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
