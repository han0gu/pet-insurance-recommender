from langchain_core.documents import Document

chunk = Document(
    page_content=('- 7. 치석제거 및 치과치료비용(부정 교합 기타 이상형성의 개선치료 포함)\n'
 '- 8. 건강식품, 보조식품, 보조치료제 및 Supplement 비용(치료를 목적으로 하는지 불문합니다.)\n'
 '- 9. 목욕 비용(약용 및 처방샴푸 값 포함) 및 벼룩, 진드기, 모낭충의 제거 비용\n'
 '- 10. 한방 및 한약(보상하는 상해 또는 질병의 치료를 위한 침술 및 물리치료는 제외), 온천요법, 산\n'
 '- 소요법, 면역요법 등의 대체적 처치에 의한 치료를 위한 비용'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
