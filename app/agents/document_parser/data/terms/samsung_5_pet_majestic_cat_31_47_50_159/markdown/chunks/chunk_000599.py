from langchain_core.documents import Document

chunk = Document(
    page_content=('제3자는 동물병원 소속 수의사 중에 정하며, 보험금 지급사유 판정에 드는 의료비용은 회\n'
 '사가 전액 부담합니다.# 제3조 (이물제거(내시경) 및 이물제거(구토유발약물)의 정의)① 이 특별약관에서 「이물제거(내시경)」 이라 함은 '
 '반려동물의 위장 등 내부의 이물질을\n'
 '제거하기 위하여 수술을 동반하지 않고 내시경 및 내시경포셉을 이용하여 비침습적으\n'
 '로 시행하는 의료행위를 말합니다.\n'
 '② 이 특별약관에서 「이물제거(구토유발약물)」 이라 함은 반려동물의 위장 등 내부의 이'),
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
