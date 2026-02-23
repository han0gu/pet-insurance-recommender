from langchain_core.documents import Document

chunk = Document(
    page_content=('| 1) 척추(등뼈)에 심한 운동장해를 남긴 때 | 40 |\n'
 '| 2) 척추(등뼈)에 뚜렷한 운동장해를 남긴 때 | 30 |\n'
 '| 3) 척추(등뼈)에 약간의 운동장해를 남긴 때 | 10 |\n'
 '| 4) 척추(등뼈)에 심한 기형을 남긴 때 | 50 |\n'
 '| 5) 척추(등뼈)에 뚜렷한 기형을 남긴 때 | 30 |\n'
 '| 6) 척추(등뼈)에 약간의 기형을 남긴 때 | 15 |\n'
 '| 7) 추간판탈출증으로 인한 심한 신경 장해 | 20 |\n'
 '| 8) 추간판탈출증으로 인한 뚜렷한 신경 장해 | 15 |'),
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
