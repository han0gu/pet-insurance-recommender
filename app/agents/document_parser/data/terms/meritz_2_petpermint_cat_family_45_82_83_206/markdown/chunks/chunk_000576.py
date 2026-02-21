from langchain_core.documents import Document

chunk = Document(
    page_content=('- 말한다.\n'
 '- 13) “뼈에 기형을 남긴 때”라 함은 상완골 또는 요골\n'
 '- 과 척골에 변형이 남아 정상에 비해 부정유합된 각\n'
 '- 변형이 15° 이상인 경우를 말한다.\n'
 '# 다. 지급률의 결정1) 한 팔의 3대 관절중 관절 하나에 기능장해가 생기고\n'
 '다른 관절 하나에 기능장해가 발생한 경우 지급률은\n'
 '각각 적용하여 합산한다.1922) 1상지(팔과 손가락)의 장해 지급률은 원칙적으로 각각\n'
 '합산하되, 지급률은 60% 한도로 한다.# 9. 다리의 장해# 가. 장해의 분류| 장해의 분류 | 지급률 |\n'
 '| --- | --- |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
