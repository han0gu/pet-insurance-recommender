from langchain_core.documents import Document

chunk = Document(
    page_content=('rowspan="4">음식물 섭취</td><td>1) 입으로 식사를 전혀 할 수 없어 계속적으로 튜브(비위 관 또는 위루관)나 경정맥 '
 '수액을 통해 부분 혹은 전 20% 적인 영양공급을 받는 상태</td><td>별 표</td></tr><tr><td>2) 수저 사용이 '
 '불가능하여 다른 사람의 계속적인 도움이 15% 없이는 식사를 전혀 할 수 없는 상태</td><td>법 '
 'ㆍ</td></tr><tr><td>3) 숟가락 사용은 가능하나 젓가락 사용이 불가능하여 음 식물 섭취에 있어 부분적으로 다른 사람의 '
 '도움이 필 10% 요한'),
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
