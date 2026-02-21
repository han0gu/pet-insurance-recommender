from langchain_core.documents import Document

chunk = Document(
    page_content=('외, 검사비포함)(재가입형) 특별약관을 따르며, 4-1. 반려견 의료비(치과및구강질환포함)(\n'
 '수술당일제외, 검사비포함)(재가입형) 특별약관에서 정하지 않은 사항은 특별약관 일반사'),
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
