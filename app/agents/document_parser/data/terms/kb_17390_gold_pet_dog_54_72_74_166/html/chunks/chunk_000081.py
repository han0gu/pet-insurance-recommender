from langchain_core.documents import Document

chunk = Document(
    page_content=('사망으로</td><td>인하여 민법에서 정한 상속순서에 따라 상속이 되는자를</td></tr></tbody></table><br><p '
 "id='97' data-category='paragraph' style='font-size:14px'>말합니다.<br>※ "
 "상속순위</p><br><h1 id='98' style='font-size:14px'>① 피상속인의</h1><br><p id='99' "
 "data-category='paragraph' style='font-size:14px'>직계비속 ② 피상속인의 직계존속</p><br><p"),
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
