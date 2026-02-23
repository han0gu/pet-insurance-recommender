from langchain_core.documents import Document

chunk = Document(
    page_content=('| 6) 한 다리의 3대 관절 중 관절 하나의 기능에 약간의 장해를 남긴 때 | 5 |\n'
 '| 7) 한 다리에 가관절이 남아 뚜렷한 장해를 남긴 때 | 20 |\n'
 '| 8) 한 다리에 가관절이 남아 약간의 장해를 남긴 때 | 10 |\n'
 '| 9) 한 다리의 뼈에 기형을 남긴 때 | 5 |\n'
 '| 10) 한 다리가 5cm 이상 짧아지거나 길어진 때 | 30 |\n'
 '| 11) 한 다리가 3cm 이상 짧아지거나 길어진 때 | 15 |\n'
 '| 12) 한 다리가 1cm 이상 짧아지거나 길어진 때 | 5 |\n'
 '- \n'
 '# 나. 장해판정기준- 1) 골절부에'),
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
