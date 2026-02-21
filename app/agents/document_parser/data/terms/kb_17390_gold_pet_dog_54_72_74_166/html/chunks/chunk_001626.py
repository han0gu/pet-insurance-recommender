from langchain_core.documents import Document

chunk = Document(
    page_content=('5</td></tr><tr><td>5) 한 발의 5개 발가락 모두의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 '
 '때</td><td>20</td></tr><tr><td>6) 한 발의 첫째 발가락의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 '
 '때</td><td>8</td></tr><tr><td>7) 한 발의 첫째 발가락 이외의 발가락의 발가락뼈 일부를 잃었을 때 또는 뚜렷한 '
 "장해를 남긴 때(발가락 하나마다)</td><td>3</td></tr></tbody></table><h1 id='146'"),
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
