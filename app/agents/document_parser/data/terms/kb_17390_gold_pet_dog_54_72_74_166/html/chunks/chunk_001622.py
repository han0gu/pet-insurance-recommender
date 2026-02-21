from langchain_core.documents import Document

chunk = Document(
    page_content=('bottom-right:(1423,481)" /></figure></td></tr></tbody></table><br><p '
 "id='138' data-category='paragraph' style='font-size:14px'>부 가 설 명</p><br><p "
 "id='139' data-category='paragraph' style='font-size:14px'>손가락</p><figure "
 'id=\'140\'><img alt="" data-coord="top-left:(836,579); '
 'bottom-right:(1425,972)"'),
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
