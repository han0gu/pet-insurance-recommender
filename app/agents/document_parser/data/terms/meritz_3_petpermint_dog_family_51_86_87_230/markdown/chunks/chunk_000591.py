from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 1) 두눈이 멀었을 때 | 100 |\n'
 '| 2) 한눈이 멀었을 때 | 50 |\n'
 '| 3) 한눈의 교정시력이 0.02 이하로 된 때 | 35 |\n'
 '| 4) 한 눈의 교정시력이 0.06 이하로 된 때 | 25 |\n'
 '| 5) 한 눈의 교정시력이 0.1 이하로 된 때 | 15 |\n'
 '| 6) 한 눈의 교정시력이 0.2 이하로 된 때 | 5 |\n'
 '| 7) 한눈의 안구(눈동자)에 뚜렷한 운동장해나 뚜렷한 조절기능장해를 남긴 때 | 10 |\n'
 '| 8) 한 눈에 뚜렷한 시야장해를 남긴 때 | 5 |'),
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
