from langchain_core.documents import Document

chunk = Document(
    page_content=('| 3) 한손의 첫째 손가락 이외의 손가락을 잃었을 때 (손가락 하나마다) | 10 |\n'
 '| 4) 한손의 5개손가락 모두의 손가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 | 30 |\n'
 '| 5) 한손의 첫째 손가락의 손가락뼈 일부를 잃었을 | 10 |\n'
 '220| 장해의 분류 | 지급률 |\n'
 '| --- | --- |\n'
 '| 때 또는 뚜렷한 장해를 남긴 때 6) 한손의 첫째 손가락 이외의 손가락의 손가락뼈 일부를 잃었을 때 또는 뚜렷한 장해를 남긴 때 '
 '(손가락 하나마다) | 5 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
