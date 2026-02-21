from langchain_core.documents import Document

chunk = Document(
    page_content=(". 장해의 분류</h1><br><table id='170' style='font-size:14px'><thead><tr><td>장해의 "
 '분류</td><td>지급률</td></tr></thead><tbody><tr><td>1) 두 눈이 멀었을 '
 '때</td><td>100</td></tr><tr><td>2) 한 눈이 멀었을 때</td><td>50</td></tr><tr><td>3) '
 '한 눈의 교정시력이 0.02 이하로 된 때</td><td>35</td></tr><tr><td>4) 한 눈의 교정시력이 0.06 이하로 된 '
 '때 5) 한 눈의'),
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
