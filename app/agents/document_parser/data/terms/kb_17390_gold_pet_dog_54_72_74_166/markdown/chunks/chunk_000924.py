from langchain_core.documents import Document

chunk = Document(
    page_content=('| 2) 흉복부장기 또는 비뇨생식기 기능을 잃었을 때 | 75 |\n'
 '| 3) 흉복부장기 또는 비뇨생식기 기능에 심한 장해를 남긴 때 | 50 |\n'
 '| 4) 흉복부장기 또는 비뇨생식기 기능에 뚜렷한 장해를 남긴 때 | 30 |\n'
 '# 5) 흉복부장기 또는 비뇨생식기 기능에 약간의 장해를 남긴 때15나.- 장해의 판정기준\n'
 '- 1) ‘심장 기능을 잃었을 때’라 함은 심장 이식을 한 경우를 말한다.\n'
 '- 2) ‘흉복부장기 또는 비뇨생식기 기능을 잃었을 때’라 함은 아래의 경우 중\n'
 '- 하나에 해당하는 때를 말한다.'),
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
