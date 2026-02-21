from langchain_core.documents import Document

chunk = Document(
    page_content=('적으로 2회 이상 입원한 경우 이를 1회 입원으로 보아 각 입원일수를 더합니다. 해\n'
 '그러나, 동일한 질병에 대한 입원이라도 반려동물 위탁비용이 지급된 최종 입원 및\n'
 '의 퇴원일로부터 180일이 지나서 개시한 입원은 새로운 입원으로 봅니다. 다만, 질\n'
 '아래와 같이 반려동물 위탁비용이 지급된 최종입원일부터 180일이 경과하도록 퇴 병\n'
 '원없이 계속 입원중인 경우에는 반려동물 위탁비용이 지급된 최종입원일의 그 다\n'
 '음날을 퇴원일로 봅니다.\n'
 '반예 시반려동물 위탁비용이![image](/image/placeholder)'),
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
