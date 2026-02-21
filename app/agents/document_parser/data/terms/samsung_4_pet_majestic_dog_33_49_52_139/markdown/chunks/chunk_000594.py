from langchain_core.documents import Document

chunk = Document(
    page_content=('따라 재가입하는 경우 또는 4-1. 반려견 의료비(치과및구강질환포함)(수술당일제외,\n'
 '검사비포함)(재가입형) 특별약관 제27조 (특별약관의 재가입에 관한 사항) 제5항에 따\n'
 '라 보험계약이 연장된 경우에는 보장개시일(책임개시일)은 이 특별약관의 보험계약일\n'
 '로 봅니다.# 제 2조 (보험금 지급에 관한 세부규정)보험수익자와 회사가 제1조(보험금의 지급사유)의 보험금 지급사유에 대해 합의하지 '
 '못\n'
 '할 때는 보험수익자와 회사가 함께 제3자를 정하고 그 제3자의 의견에 따를 수 있습니다.'),
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
