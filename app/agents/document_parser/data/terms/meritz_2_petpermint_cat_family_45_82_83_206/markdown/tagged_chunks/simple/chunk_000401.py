from langchain_core.documents import Document

chunk = Document(
    page_content=('- ⑦ 미용으로 인한 비용\n'
 '- ⑧ 귀 성형, 꼬리 성형, 성대제거 및 미용성형을 위한\n'
 '- 수술 및 처치에 따른 비용\n'
 '- ⑨ 손톱절제(며느리발톱 제거 포함), 잔존유치, 잠복고\n'
 '- 환, 제대허니아(배꼽부위탈장), 항문낭 제거 등 건\n'
 '150강동물에 실시하는 외과수술 및 기타 검사 또는 점\n'
 '안, 귀청소 등의 관리 비용- ⑩ 첩모난생(속눈썹 질환), 눈물샘으로 인한 비용\n'
 '- ⑪ 입원중의 식이(食餌)에 해당하지 않는 음식물 및 식\n'
 '- 이요법, 수의사 처방 의약품 이외의 것(건강보조 식\n'
 '- 품, 의약품지정이 되어 있지 않은 한방약, 의약부외'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000401',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
