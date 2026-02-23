from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 분류</p><br><p id='127' data-category='paragraph' "
 "style='font-size:14px'>별</p><p id='128' data-category='paragraph' "
 "style='font-size:16px'>장해의 분류 지급률<br>1) 한 손의 5개 손가락을 모두 잃었을 때 55<br>2) 한 손의 "
 '첫째 손가락을 잃었을 때 15<br>3) 한 손의 첫째 손가락 이외의 손가락을 잃었을 때 법<br>10 ㆍ<br>(손가락 '
 '하나마다)<br>4) 한 손의 5개 손가락 모두의 손가락뼈 일부를'),
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
