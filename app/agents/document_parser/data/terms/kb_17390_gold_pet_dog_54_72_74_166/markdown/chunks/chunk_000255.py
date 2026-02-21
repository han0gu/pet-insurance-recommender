from langchain_core.documents import Document

chunk = Document(
    page_content=('- 하여 지급합니다. 다만, 장해분류표의 각 신체부위별 판정기준에 별도로 정한 경\n'
 '- 우에는 그 기준에 따릅니다.\n'
 '- \uf000 다른 상해로 인하여 후유장해가 2회 이상 발생하였을 경우에는 그 때마다 이에 해\n'
 '- 당하는 후유장해지급률을 결정합니다. 그러나 그 후유장해가 이미 후유장해보험\n'
 '- 금을 지급받은 동일한 부위에 가중된 때에는 최종 장해상태에 해당하는 후유장해\n'
 '- 보험금에서 이미 지급받은 후유장해보험금을 차감하여 지급합니다. 다만, 장해분\n'
 '- 류표의 각 신체부위별 판정기준에서 별도로 정한 경우에는 그 기준에 따릅니다.'),
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
