from langchain_core.documents import Document

chunk = Document(
    page_content=('⑥ 중성화, 불임 및 피임을 목적으로 한 수술 및 처치에 따른 비용 ⑦ 미용으로 인한 비용 ⑧ 귀 성형, 꼬리 성형, 성대제거 및 '
 '미용성형을 위한 수술 및 처치에 따른 비용 ⑨ 손톱절제(며느리발톱 제거 포함), 잔존유치, 잠복고 환, 제대허니아(배꼽부위탈장), 항문낭 '
 '제거 등 건 강동물에 실시하는 외과수술 및 기타 검사 또는 점 안, 귀청소 등의 관리 비용 ⑩ 첩모난생(속눈썹 질환), 눈물샘으로 인한 '
 '비용 ⑪ 입원중의 식이(食餌)에 해당하지 않는 음식물 및 식 이요법, 수의사 처방 의약품 이외의 것(건강보조 식 품, 의약품지정이'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 158},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['dental', 'skin', 'eye', 'other']},
 'indexing': {'chunk_id': 'chunk_000513',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
