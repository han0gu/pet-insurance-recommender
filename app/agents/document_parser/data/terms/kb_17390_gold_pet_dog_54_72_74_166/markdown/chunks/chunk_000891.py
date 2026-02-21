from langchain_core.documents import Document

chunk = Document(
    page_content=('| 6) 한 팔의 3대 관절 중 관절 하나의 기능에 약간의 장해를 남긴 때 | 5 |\n'
 '| 7) 한 팔에 가관절이 남아 뚜렷한 장해를 남긴 때 | 20 특별 |\n'
 '| 8) 한 팔에 가관절이 남아 약간의 장해를 남긴 때 | 10 약 관 |\n'
 '| 9) 한 팔의 뼈에 기형을 남긴 때 | 5 |\n'
 '# 나. 장해판정기준- 1) 골절부에 금속내고정물 등을 사용하였기 때문에 그것이 기능장해의 원인\n'
 '- 별\n'
 '- 이 되는 때에는 그 내고정물 등이 제거된 후 장해를 평가한다. 단, 제거\n'
 '- 표'),
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
