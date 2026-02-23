from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 1) 두다리의 발목이상을 잃었을 때 | 100 |\n'
 '| 2) 한다리의 발목이상을 잃었을 때 | 60 |\n'
 '| 3) 한다리의 3대관절중 관절 하나의 기능을 완전히 잃었 을 때 | 30 |\n'
 '| 4) 한다리의 3대관절중 관절 하나의 기능에 심한 장해 를 남긴 때 | 20 |\n'
 '| 5) 한다리의 3대관절중 관절 하나의 기능에 뚜렷한 장해 를 남긴 때 | 10 |\n'
 '| 6) 한다리의 3대관절중 관절 하나의 기능에 약간의 장해 를 남긴 때 | 5 |\n'
 '| 7) 한다리에 가관절이 남아 뚜렷한 장해를 남긴 때 | 20 |'),
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
