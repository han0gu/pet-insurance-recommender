from langchain_core.documents import Document

chunk = Document(
    page_content=('| 음식물 섭취 | 2) 수저 사용이 불가능하여 다른 사람의 계속적인 도움이 15% 없이는 식사를 전혀 할 수 없는 상태 | 법 ㆍ |\n'
 '| 음식물 섭취 | 3) 숟가락 사용은 가능하나 젓가락 사용이 불가능하여 음 식물 섭취에 있어 부분적으로 다른 사람의 도움이 필 10% '
 '요한 상태 | 규정 |\n'
 '| 음식물 섭취 | 4) 독립적인 음식물 섭취는 가능하나 젓가락을 이용하여 5% 생선을 바르거나 음식물을 자르지는 못하는 상태 |  |'),
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
