from langchain_core.documents import Document

chunk = Document(
    page_content=('신됩니다.# 제9조 (준용규정)이 특별약관에 정하지 않은 사항은 특별약관 일반사항을 따릅니다. 특별약관 일반사항에\n'
 '서도 정하지 않은 사항은 보통약관을 따릅니다. 다만, 보통약관 제5조(보험금을 지급하지\n'
 '않는 사유), 제10조(환급금의 중도인출), 제11조(만기환급금의 지급)은 제외합니다.- 127 -제도성 특별약관※ 약관에서 인용된 '
 '법·규정은 「별표 및 참고」 의 「약관에서 인용된 법·규정」 에서'),
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
