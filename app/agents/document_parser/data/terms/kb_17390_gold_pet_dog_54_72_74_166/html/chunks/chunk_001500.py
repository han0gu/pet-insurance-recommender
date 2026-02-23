from langchain_core.documents import Document

chunk = Document(
    page_content=("id='190' data-category='paragraph' style='font-size:14px'>검사(ABR), 이음향방사검사’ "
 "등을 추가실시 후 장해를 평가한다.</p><br><p id='191' data-category='paragraph' "
 "style='font-size:14px'>다"),
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
