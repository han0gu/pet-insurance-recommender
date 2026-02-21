from langchain_core.documents import Document

chunk = Document(
    page_content=("id='232' data-category='paragraph' style='font-size:14px'>해</p><figure "
 'id=\'233\'><img alt="" data-coord="top-left:(1480,439); '
 'bottom-right:(1501,485)" /></figure><p id=\'0\' data-category=\'paragraph\' '
 "style='font-size:14px'>에서 그 권리를 취득합니다.</p><br><p id='1' data-category='list' "
 "style='font-size:14px'>1"),
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
