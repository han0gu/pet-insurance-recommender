from langchain_core.documents import Document

chunk = Document(
    page_content=('수술적 치료를 시행하였음에도 불구하고 골절부의 유합이 이루어지 사항\n'
 '지 않는 ‘불유합’ 상태를 말하며, 골유합이 지연되는 지연유합은- 제외한다.\n'
 '- 13) ‘가관절이 남아 약간의 장해를 남긴 때’라 함은 경골과 종아리뼈 중 어\n'
 '- 느 한 뼈에 가관절이 남은 경우를 말한다. 보\n'
 '- 14) ‘뼈에 기형을 남긴 때’라 함은 대퇴골 또는 경골에 기형이 남아 정상에 통약\n'
 '- 비해 부정유합된 각 변형이 15° 이상인 경우를 말한다.\n'
 '- 관\n'
 '- 15) 다리 길이의 단축 또는 과신장은 스캐노그램(scanogram)을 통하여 측정\n'
 '한다.-'),
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
