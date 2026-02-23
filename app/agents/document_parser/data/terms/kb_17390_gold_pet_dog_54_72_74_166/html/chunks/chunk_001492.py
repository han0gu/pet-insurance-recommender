from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다만, 안면부의 추상(추한 모습)은 두 가지 장해평가</p><br><h1 id='180' "
 "style='font-size:16px'>방법 중 피보험자에게 유리한 것을 적용한다.</h1><p id='181' "
 "data-category='paragraph' style='font-size:16px'>2. 귀의 장해</p><table id='182' "
 "style='font-size:16px'><thead></thead><tbody><tr><td>가"),
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
