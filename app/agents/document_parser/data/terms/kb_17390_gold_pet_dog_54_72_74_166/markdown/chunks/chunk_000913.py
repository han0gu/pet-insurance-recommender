from langchain_core.documents import Document

chunk = Document(
    page_content=('- 3) 손가락에는 첫째 손가락에 2개의 손가락관절이 있다. 그 중 심장에서 가\n'
 '- 까운 쪽부터 중수지관절, 지관절이라 한다.\n'
 '- 4) 다른 네 손가락에는 3개의 손가락관절이 있다. 그 중 심장에서 가까운 쪽\n'
 '- 부터 중수지관절, 제1지관절(근위지관절) 및 제2지관절(원위지관절)이\n'
 '- 라 부른다.\n'
 '- 5) ‘손가락을 잃었을 때’라 함은 첫째 손가락에서는 지관절부터 심장에서\n'
 '- 가까운 쪽에서, 다른 네 손가락에서는 제1지관절(근위지관절)부터(제1지\n'
 '- 관절 포함) 심장에서 가까운 쪽으로 손가락이 절단되었을 때를 말한다.'),
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
