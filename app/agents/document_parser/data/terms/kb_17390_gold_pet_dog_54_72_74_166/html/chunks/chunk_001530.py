from langchain_core.documents import Document

chunk = Document(
    page_content=(". 외모의 추상(추한 모습)장해</h1><p id='4' data-category='list'></p><br><p id='5' "
 "data-category='paragraph' style='font-size:14px'>가"),
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
