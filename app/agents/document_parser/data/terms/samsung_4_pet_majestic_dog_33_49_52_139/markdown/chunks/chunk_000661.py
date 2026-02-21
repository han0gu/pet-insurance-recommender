from langchain_core.documents import Document

chunk = Document(
    page_content=('- 급금 산출방법서" 에 정하는 바에 따라 회사가 적립한 사망당시 이 특별약관의 계약\n'
 '- 자적립액 및 미경과보험료를 계약자에게 지급하고, 이 특별약관은 더 이상 효력이 없\n'
 '- 습니다.\n'
 '- ② 보험의 목적이 다수인 경우 제1항은 보험의 목적별로 각각 적용합니다.\n'
 '# 제13조 (특별약관의 자동갱신)이 특별약관은 제도성 특별약관 5-1. [갱신형] 특별약관의 자동갱신 특별약관에 따라 갱\n'
 '신됩니다.# 제14조 (준용규정)이 특별약관에 정하지 않은 사항은 4-1. 반려견 의료비(치과및구강질환포 함)(수술당일제'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
