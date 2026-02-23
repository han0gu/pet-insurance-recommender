from langchain_core.documents import Document

chunk = Document(
    page_content=('- 2) 1하지(다리와 발가락)의 후유장해 지급률은 원칙적으로 각각 합산하되, 지급률\n'
 '- 은 60% 한도로 한다.\n'
 '# 10. 손가락의 장해# 가. 장해의 분류| 장 해 의 분 류 | 지급률(%) |\n'
 '| --- | --- |\n'
 '| 1) 한 손의 5개 손가락을 모두 잃었을 때 | 55 |\n'
 '| 2) 한 손의 첫째 손가락을 잃었을 때 | 15 |\n'
 '| 3) 한 손의 첫째 손가락 이외의 손가락을 잃었을 때(손가락 하나마다) | 10 |\n'
 '| 4) 한 손의 5개 손가락 모두의 손가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 30 |'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
