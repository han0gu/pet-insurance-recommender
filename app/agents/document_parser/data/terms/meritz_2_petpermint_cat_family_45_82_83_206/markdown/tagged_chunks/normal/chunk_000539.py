from langchain_core.documents import Document

chunk = Document(
    page_content=('- 다) 위‧아래턱(상ㆍ하악)의 부정교합(전방, 측방)이\n'
 '- 1cm이상인 경우\n'
 '- 라) 양측 각 1개 또는 편측 2개 이하의 치아만 교합\n'
 '- 되는 상태\n'
 '- 마) 연하기능검사(비디오 투시검사)상 연하장애가\n'
 '- 있고, 유동식 섭취시 간헐적으로 흡인이 발생\n'
 '- 하고 부드러운 고형식 외에는 섭취가 불가능한\n'
 '- 상태\n'
 '5) 개구(입을 벌림)장해는 턱관절의 이상으로 개구(입\n'
 '을 벌림)운동 제한이 있는 상태를 말하며, 최대 개182구(입을 벌림)상태에서 위‧아래턱(상ㆍ하악)의 가운\n'
 '데 앞니(중절치)간 거리를 기준으로 한다. 단, 가'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other',
            'risk_domains': ['dental', 'digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000539',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
