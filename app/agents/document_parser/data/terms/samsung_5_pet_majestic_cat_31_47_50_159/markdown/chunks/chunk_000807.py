from langchain_core.documents import Document

chunk = Document(
    page_content=('- 뼈에 가관절이 남은 경우를 말한다.\n'
 '- 14) "뼈에 기형을 남긴 때" 라 함은 대퇴골 또는 경골에 기형이 남아 정상에 비해\n'
 '- - 144 -\n'
 '부정유합된 각 변형이 15° 이상인 경우를 말한다.15) 다리 길이의 단축 또는 과신장은 스캐노그램(scanogram)을 통하여 '
 '측정한다.# 다. 지급률의 결정- 1) 한 다리의 3대 관절 중 관절 하나에 기능장해가 생기고 다른 관절 하나에 기능\n'
 '- 장해가 발생한 경우 지급률은 각각 적용하여 합산한다.\n'
 '- 2) 1하지(다리와 발가락)의 후유장해 지급률은 원칙적으로 각각 합산하되, 지급률'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
