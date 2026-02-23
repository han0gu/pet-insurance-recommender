from langchain_core.documents import Document

chunk = Document(
    page_content=('나누어 지<br>급할 금액을 일시에 지급하는 경우에는 평균공시이율을 연단위 복리로 할인한 금액<br>을 지급합니다.</p><p '
 "id='89' data-category='paragraph' style='font-size:14px'>58 KB 금쪽같은 "
 "펫보험(강아지)(무배당)(26.01)</p><br><table id='90'"),
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
