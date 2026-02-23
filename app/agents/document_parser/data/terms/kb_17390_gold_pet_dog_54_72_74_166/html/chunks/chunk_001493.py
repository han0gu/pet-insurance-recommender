from langchain_core.documents import Document

chunk = Document(
    page_content=('. 장해의 분류</td><td>보</td></tr><tr><td>장해의 분류</td><td>지급률 '
 '통약</td></tr><tr><td>1) 두 귀의 청력을 완전히 잃었을 때 80 2) 한 귀의 청력을 완전히 잃고, 다른 귀의 청력에 '
 '심한 장해를 남긴 때</td><td>관 45</td></tr><tr><td>3) 한 귀의 청력을 완전히 잃었을 '
 '때</td><td>25</td></tr><tr><td>4) 한 귀의 청력에 심한 장해를 남긴 때</td><td>15 특별 '
 '약</td></tr><tr><td>5) 한 귀의 청력에 약간의 장해를'),
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
