from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 분류<br>장해의 분류 지급률<br>1) 코의 호흡기능을 완전히 잃었을 때 15</p><br><h1 id='202' "
 "style='font-size:14px'>2) 코의 후각기능을 완전히 잃었을 때</h1><br><p id='203' "
 "data-category='paragraph' style='font-size:14px'>5</p><h1 id='204' "
 "style='font-size:14px'>나.</h1><br><table id='205'"),
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
