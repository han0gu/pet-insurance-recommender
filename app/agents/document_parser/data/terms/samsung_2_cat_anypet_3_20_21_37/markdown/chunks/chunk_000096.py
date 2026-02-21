from langchain_core.documents import Document

chunk = Document(
    page_content=('- 발생한 손해\n'
 '- 4. 수의사의 치료상의 과오로 생긴 손해, 수의사 자격이 없는 자의 치료행위로 인한 손해\n'
 '- 5. 지진, 분화, 해일, 홍수 또는 이와 비슷한 천재지변\n'
 '- 6. 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소요, 기타 이들과 유사한 사태\n'
 '- 7. 핵연료물질(사용이 끝난 연료를 포함합니다. 이하 같습니다) 또는 핵연료 물질에 의하여 오염된 물\n'
 '- 질(원자핵분열 생성물을 포함합니다)의 방사성, 폭발성 또는 그 밖의 유해한 특성에 의한 사고\n'
 '- 8. 위 제7호 이외의 방사선을 쬐는 것 또는 방사능 오염'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
