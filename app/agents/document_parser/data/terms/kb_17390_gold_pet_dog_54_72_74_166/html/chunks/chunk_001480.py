from langchain_core.documents import Document

chunk = Document(
    page_content=('남긴 때</td><td>5</td></tr><tr><td>9) 한 눈의 눈꺼풀에 뚜렷한 결손을 남긴 '
 '때</td><td>10</td></tr><tr><td>10) 한 눈의 눈꺼풀에 뚜렷한 운동장해를 남긴 '
 "때</td><td>5</td></tr></tbody></table><p id='171' data-category='paragraph' "
 "style='font-size:16px'>- 140 -</p><p id='172' data-category='paragraph' "
 "style='font-size:16px'>나"),
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
