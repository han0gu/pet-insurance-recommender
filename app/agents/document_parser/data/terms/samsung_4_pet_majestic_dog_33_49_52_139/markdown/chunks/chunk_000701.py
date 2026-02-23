from langchain_core.documents import Document

chunk = Document(
    page_content=('- 등으로 갱신됩니다. 다만, 계약자는 갱신일 현재의 약관 등에 대해 갱신일로부터 90\n'
 '- 일 이내에 그 계약을 취소할 수 있으며, 이 경우 회사는 갱신일 이후 계약자가 납입\n'
 '- 한 해당 갱신계약의 보험료를 돌려 드립니다.\n'
 '- 제7조 (준용규정)\n'
 '이 특별약관에 정하지 않은 사항은 보통약관 및 해당 갱신계약을 따릅니다.- 131 -# 5-2. 이륜자동차 운전 및 탑승 중 상해 부담보 '
 '특별약관# 제1조 (특별약관의 체결 및 효력)- ① 이 특별약관은 보험계약(특별약관이 부가된 경우에는 그 특별약관을 포함합니다. 이하'),
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
