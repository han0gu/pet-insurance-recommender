from langchain_core.documents import Document

chunk = Document(
    page_content=('길이 3.0cm 이상 ~ 5.0cm 미만</td><td>S0037</td></tr><tr><td>4) 길이 5.0cm 이상 ~ 7.5cm '
 '미만</td><td>S0038</td></tr><tr><td>5) 길이 7.5cm 이상 ~ 10.0cm '
 '미만</td><td>S0039</td></tr><tr><td>주: 길이 10cm이상 창상봉합술을 시행할경우 소 정점수에 52.00점을 '
 '가산하며, 창상봉합 길 이가 5cm 증가될때마다 52.00점을 추가 가산 '
 '한다.</td><td>S0040</td></tr><tr><td>(2) 변연절제를'),
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
