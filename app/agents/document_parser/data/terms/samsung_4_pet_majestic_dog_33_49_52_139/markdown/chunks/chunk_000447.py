from langchain_core.documents import Document

chunk = Document(
    page_content=('- 라. 투견, 경주견 등 흥행을 목적으로 사육·관리하는 개(犬)\n'
 '<용어풀이># [흥행]영리를 목적으로 연극, 영화, 서커스 등을 요금을 받고 대중에게 보여주는 행위를 말합니다.마. 유기동물 보호센터 '
 '등에서 사육·관리하는 개(犬)# ② 지급사유 관련 용어1. 상해: 보험기간 중에 발생한 급격하고도 우연한 외래의 사고로 반려견에 입은 '
 '상해\n'
 '를 말하며, 유독 가스 또는 유독 물질을 반려견이 우연히 일시적으로 흡입, 흡수\n'
 '또는 섭취한 결과로 생긴 중독 증상을 포함합니다. 그러나 음식물 섭취로 인한 증'),
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
