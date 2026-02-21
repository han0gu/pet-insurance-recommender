from langchain_core.documents import Document

chunk = Document(
    page_content=('. (나) 근육에 달하는것</td><td>SC030</td></tr><tr><td>1) 길이 2.5cm '
 '미만</td><td>SC031</td></tr><tr><td>2) 길이 2.5cm 이상 ~ 5.0cm '
 '미만</td><td>SC032</td></tr><tr><td>3) 길이 5.0cm 이상 ~ 10.0cm '
 '미만</td><td>SC039</td></tr><tr><td></td><td></td></tr><tr><td>주: 길이 10cm이상 '
 '창상봉합술을 시행할경우 소 정점수에 103.14점을 가산하며, 창상봉합 길 이가 10cm'),
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
