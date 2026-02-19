from langchain_core.documents import Document

chunk = Document(
    page_content=('⑩ 첩모난생(속눈썹 질환), 눈물샘으로 인한 비용 ⑪ 입원중의 식이(食餌)에 해당하지 않는 음식물 및 식 이요법, 수의사 처방 의약품 '
 '이외의 것(건강보조 식 품, 의약품지정이 되어 있지 않은 한방약, 의약부외 품 등) ⑫ 한의학(단, 침술을 제외합니다.), 인도 의학, '
 '허브요 법, 아로마테라피 등의 대체의료, 재활치료 ⑬ 목욕비용(약욕 및 처방샴푸 값 포함) 및 귀 세정제 (이어 클리너), 예방 가능한 '
 '기생충(벼룩, 진드기, 모낭충 등)의 제거비용 및 기생충으로 발생한 질병의 치료비 ⑭ 반려동물호텔 또는 보관 비용, 산책료,'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 151},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['eye']},
 'indexing': {'chunk_id': 'chunk_000507',
              'chunk_char_len': 296,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
