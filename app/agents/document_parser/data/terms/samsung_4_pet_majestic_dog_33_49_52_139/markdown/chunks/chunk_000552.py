from langchain_core.documents import Document

chunk = Document(
    page_content=('- 강질환포함)(수술당일제외, 검사비포함)(재가입형) 특별약관의 자기부담금을 말합니다\n'
 '- ⑤ 회사가 지급할 제1항에서 정한 보험금은 피보험자가 부담한 당일 발생한 의료비에서\n'
 '- 제4항에서 정한「자기부담금」및 제3항에서 정한「반려견의료비(치과및구강질환포함\n'
 '- )(수술당일제외,검사비포함)보험금의 1일 한도」를 차감한 후 보상비율을 곱한 금액으\n'
 '- 로 제1항에서 정한 보상한도액을 한도로 합니다. 단, 4-1. 반려견 의료비(치과및구강\n'
 '- 질환포함)(수술당일제외, 검사비포함)(재가입형) 특별약관의 보험금이 1일당 보상한도'),
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
