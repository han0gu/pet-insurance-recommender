from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 1) 두 눈이 멀었을 때 | 100 |\n'
 '| 2) 한 눈이 멀었을 때 | 50 |\n'
 '| 3) 한 눈의 교정시력이 0.02 이하로 된 때 | 35 |\n'
 '| 4) 한 눈의 교정시력이 0.06 이하로 된 때 5) 한 눈의 교정시력이 0.1 이하로 된 때 | 25 15 |\n'
 '| 6) 한 눈의 교정시력이 0.2 이하로 된 때 | 5 |\n'
 '|  |  |\n'
 '| 7) 한 눈의 안구(눈동자)에 뚜렷한 운동장해나 뚜렷한 조절기능 장해를 남긴 때 | 10 |\n'
 '| 8) 한 눈에 뚜렷한 시야장해를 남긴 때 | 5 |'),
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
