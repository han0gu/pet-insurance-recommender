from langchain_core.documents import Document

chunk = Document(
    page_content=("KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='158' data-category='paragraph' "
 "style='font-size:14px'>12.</p><br><h1 id='159' style='font-size:14px'>흉․복부장기 "
 "및</h1><br><h1 id='160' style='font-size:14px'>비뇨생식기의 장해</h1><br><table "
 "id='161' style='font-size:14px'><thead><tr><td>가"),
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
