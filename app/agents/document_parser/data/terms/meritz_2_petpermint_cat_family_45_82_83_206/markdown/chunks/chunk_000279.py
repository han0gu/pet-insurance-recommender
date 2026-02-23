from langchain_core.documents import Document

chunk = Document(
    page_content=('- 또는 중대한 과실\n'
 '- ② 지진, 분화, 해일, 홍수 또는 이와 유사한 자연재해로\n'
 '- 생긴 손해\n'
 '- ③ 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소\n'
 '- 요, 기타 이들과 유사한 사태\n'
 '- ④ 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의\n'
 '- 방사성, 폭발성, 그 밖의 유해한 특성 또는 이들의\n'
 '- 특성에 의한 사고\n'
 '- ⑤ 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown'},
)
