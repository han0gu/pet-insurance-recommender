from langchain_core.documents import Document

chunk = Document(
    page_content=('- 험수익자에게 지급합니다.\n'
 '- \uf000 제1항의 사망은 동물병원에서 적법하게 시행된 안락사를 포함합니다. 단, 이 경\n'
 '- 우 동물병원에서 발급한 소견서를 제출하여야 합니다.\n'
 '- \uf000 제1항의 "동물장묘업체"라 함은 동물보호법 제69조(영업의 허가)에서 정하는 동\n'
 '- 물장묘업자로써, 동물 전용의 장례식장, 동물화장시설, 동물건조장시설, 동물수\n'
 '- 분해장시설, 동물 전용의 봉안시설 중 어느 하나 이상의 시설을 설치, 운영하는\n'
 '118 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)영업을 행하는 시설을 말합니다.'),
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
