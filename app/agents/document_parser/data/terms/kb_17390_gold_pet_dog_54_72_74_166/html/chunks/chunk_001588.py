from langchain_core.documents import Document

chunk = Document(
    page_content=('남은 경우를 말한다.<br>주) 가관절이란, 충분한 경과 및 골이식술 등 골유합을 얻는데 필요한<br>수술적 치료를 시행하였음에도 '
 '불구하고 골절부의 유합이 이루어지<br>지 않는 ‘불유합’ 상태를 말하며, 골유합이 지연되는 지연유합은<br>제외한다.<br>12) '
 '‘가관절이 남아 약간의 장해를 남긴 때’라 함은 요골과 척골 중 어느<br>한 뼈에 가관절이 남은 경우를 말한다.<br>13) ‘뼈에 '
 '기형을 남긴 때’라 함은 상완골 또는 요골과 척골에 변형이 남아<br>정상에 비해 부정유합된 각 변형이 15° 이상인 경우를'),
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
