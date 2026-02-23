from langchain_core.documents import Document

chunk = Document(
    page_content=(". 다리의 장해</h1><table id='99' "
 "style='font-size:16px'><thead><tr><td>가.</td><td></td></tr></thead><tbody><tr><td>장해의 "
 '분류 장해의 분류</td><td>지급률</td></tr><tr><td>1) 두 다리의 발목 이상을 잃었을 '
 '때</td><td>100</td></tr><tr><td>2) 한 다리의 발목 이상을 잃었을 때 3) 한 다리의 3대 관절 중 관절 하나의 '
 '기능을 완전히 잃었을 때 30</td><td>60</td></tr><tr><td>4) 한'),
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
