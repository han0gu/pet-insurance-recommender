from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="11">폐</td><td>기관지염 급성 세기관지염</td><td>J21</td></tr><tr><td>달리 분류되지 않은 '
 '바이러스폐렴 폐렴연쇄알균에 의한 폐렴</td><td>J12</td></tr><tr><td>인플루엔자균에 의한 폐렴</td><td>J13 '
 'J14</td></tr><tr><td>달리 분류되지 않은 세균성 폐렴</td><td>J15</td></tr><tr><td>달리 분류되지 '
 '않은 기타 감염성 병원체에 의한'),
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
