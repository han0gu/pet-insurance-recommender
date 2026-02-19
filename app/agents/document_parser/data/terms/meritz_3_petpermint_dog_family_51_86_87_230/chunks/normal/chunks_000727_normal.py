from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 약간의 개구(입을 벌림)운동 제한 또는 약간의 저작(씹기)운동 제한으로 부드러운 고형식(밥, 빵 등)만 섭취 가능한 경우 나) '
 '위‧아래턱(상ㆍ하악)의 가운데 앞니(중절치)간 최대 개구(입을 벌림)운동이 2cm이하로 제한되 는 경우 다) 위‧아래턱(상ㆍ하악)의 '
 '부정교합(전방, 측방)이 1cm이상인 경우 라) 양측 각 1개 또는 편측 2개 이하의 치아만 교합 되는 상태 마) 연하기능검사(비디오 '
 '투시검사)상 연하장애가 있고, 유동식 섭취시 간헐적으로 흡인이 발생 하고 부드러운 고형식 외에는 섭취가 불가능한 상태'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 207},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion',
            'risk_domains': ['other', 'dental', 'digestive']},
 'indexing': {'chunk_id': 'chunk_000727',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
