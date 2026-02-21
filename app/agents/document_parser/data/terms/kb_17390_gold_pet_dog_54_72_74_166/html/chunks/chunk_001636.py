from langchain_core.documents import Document

chunk = Document(
    page_content=('. 장해의 분류</td><td></td></tr></thead><tbody><tr><td>장해의 '
 '분류</td><td>지급률</td></tr><tr><td>1) 심장 기능을 잃었을 '
 '때</td><td>100</td></tr><tr><td>2) 흉복부장기 또는 비뇨생식기 기능을 잃었을 '
 '때</td><td>75</td></tr><tr><td>3) 흉복부장기 또는 비뇨생식기 기능에 심한 장해를 남긴 '
 '때</td><td>50</td></tr><tr><td>4) 흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를 남긴'),
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
