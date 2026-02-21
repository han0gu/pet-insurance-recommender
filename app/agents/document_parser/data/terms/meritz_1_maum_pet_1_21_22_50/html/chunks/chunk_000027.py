from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 사고일 또<br>는 발병일부터 365일 이내인 경우에 한합니다.<br>‡ 제1항의 「수술」이라 함은 수의사가 치료가 필요하다고 '
 '인정한 경우로서 수의사의 관<br>리하에 치료를 직접적인 목적으로 기구를 사용하여 생체(生體)에 절단, 절제 등의 조작을 가<br>하는 '
 '것을 말합니다'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
