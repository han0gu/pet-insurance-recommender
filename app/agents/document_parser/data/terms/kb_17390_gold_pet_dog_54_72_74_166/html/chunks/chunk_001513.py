from langchain_core.documents import Document

chunk = Document(
    page_content=(". 씹어먹거나</header><br><h1 id='209' style='font-size:14px'>말하는 "
 "장해</h1><br><table id='210' style='font-size:14px'><thead><tr><td>가"),
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
