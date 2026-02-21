from langchain_core.documents import Document

chunk = Document(
    page_content=('| --- | --- |\n'
 '| 장해의 분류 | 지급률 사항 |\n'
 '| 1) 두 팔의 손목 이상을 잃었을 때 | 100 |\n'
 '| 2) 한 팔의 손목 이상을 잃었을 때 3) 한 팔의 3대 관절 중 관절 하나의 기능을 완전히 30 | 60 |\n'
 '| 잃었을 때 4) 한 팔의 3대 관절 중 관절 하나의 기능에 심한 장해를 남긴 때 20 | 보 통약 |\n'
 '| 5) 한 팔의 3대 관절 중 관절 하나의 기능에 뚜렷한 장해를 남긴 때 | 관 10 |\n'
 '| 6) 한 팔의 3대 관절 중 관절 하나의 기능에 약간의 장해를 남긴 때 | 5 |'),
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
