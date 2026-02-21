from langchain_core.documents import Document

chunk = Document(
    page_content=('래의 경우 중 하나에 해당하는 때를 말한다.- 가) 완전 강직(관절굳음)\n'
 '- 나) 근전도 검사상 완전손상(complete injury) 소견\n'
 '- 이 있으면서 도수근력검사(MMT)에서 근력이 “0\n'
 '- 등급(Zero)"인 경우\n'
 '8) “관절 하나의 기능에 심한 장해를 남긴 때”라 함은\n'
 '아래의 경우 중 하나에 해당하는 때를 말한다.- 가) 해당 관절의 운동범위 합계가 정상운동범위의\n'
 '- 1/4이하로 제한된 경우\n'
 '- 나) 인공관절이나 인공골두를 삽입한 경우\n'
 '- 다) 객관적 검사(스트레스 엑스선)상 15mm 이상의'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'joint']},
 'indexing': {'chunk_id': 'chunk_000582',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
