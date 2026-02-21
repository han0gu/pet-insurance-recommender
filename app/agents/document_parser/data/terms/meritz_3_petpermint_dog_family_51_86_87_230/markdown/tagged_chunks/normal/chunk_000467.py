from langchain_core.documents import Document

chunk = Document(
    page_content=('- 환, 제대허니아(배꼽부위탈장), 항문낭 제거 등 건\n'
 '- 강동물에 실시하는 외과수술 및 기타 검사 또는 점\n'
 '- 안, 귀청소 등의 관리 비용\n'
 '- ⑩ 첩모난생(속눈썹 질환), 눈물샘으로 인한 비용\n'
 '- ⑪ 입원중의 식이(食餌)에 해당하지 않는 음식물 및 식\n'
 '- 이요법, 수의사 처방 의약품 이외의 것(건강보조 식\n'
 '- 품, 의약품지정이 되어 있지 않은 한방약, 의약부외\n'
 '- 품 등)\n'
 '- ⑫ 한의학(단, 침술을 제외합니다.), 인도 의학, 허브요\n'
 '- 법, 아로마테라피 등의 대체의료, 재활치료'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000467',
              'chunk_char_len': 271,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
