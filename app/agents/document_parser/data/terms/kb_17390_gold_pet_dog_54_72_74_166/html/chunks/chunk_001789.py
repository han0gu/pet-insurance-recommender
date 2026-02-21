from langchain_core.documents import Document

chunk = Document(
    page_content=('알레르기성</td><td>아토피성 피부염</td><td>L20</td></tr><tr><td '
 'rowspan="2">천</td><td>혈관운동성 및 알레르기성 비염 비염</td><td>J30</td></tr><tr><td>천식 '
 '식</td><td>J45</td></tr><tr><td rowspan="2"></td><td>천식지속 '
 '상태</td><td>J46</td></tr><tr><td>급성 급성 기관지염</td><td>J20</td></tr><tr><td '
 'rowspan="11">폐</td><td>기관지염 급성'),
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
