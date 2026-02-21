from langchain_core.documents import Document

chunk = Document(
    page_content=('- 1/2 이하로 제한된 경우\n'
 '- 나) 객관적 검사(스트레스 엑스선)상 10mm 이상의\n'
 '- 동요관절(관절이 흔들리거나 움직이는 것)이 있\n'
 '- 는 경우\n'
 '- 다) 근전도 검사상 불완전한 손상(incomplete\n'
 '- injury)소견이 있으면서 도수근력검사(MMT)에\n'
 '- 서 근력이 2등급(poor)인 경우\n'
 '10) “관절 하나의 기능에 약간의 장해를 남긴 때”라\n'
 '함은 아래의 경우 중 하나에 해당하는 때를 말한다.\n'
 '가) 해당 관절의 운동범위 합계가 정상운동범위의'),
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
 'indexing': {'chunk_id': 'chunk_000584',
              'chunk_char_len': 255,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
