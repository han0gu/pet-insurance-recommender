from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 장해의 분류 | 지급률 통약 |\n'
 '| 1) 두 귀의 청력을 완전히 잃었을 때 80 2) 한 귀의 청력을 완전히 잃고, 다른 귀의 청력에 심한 장해를 남긴 때 | 관 45 '
 '|\n'
 '| 3) 한 귀의 청력을 완전히 잃었을 때 | 25 |\n'
 '| 4) 한 귀의 청력에 심한 장해를 남긴 때 | 15 특별 약 |\n'
 '| 5) 한 귀의 청력에 약간의 장해를 남긴 때 | 5 관 |\n'
 '| 6) 한 귀의 귓바퀴의 대부분이 결손된 때 | 10 |\n'
 '| 7) 평형기능에 장해를 남긴 때 | 10 |'),
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
