from langchain_core.documents import Document

chunk = Document(
    page_content=('- 에서 근력이 2등급(poor)인 경우\n'
 '10) “관절 하나의 기능에 약간의 장해를 남긴 때”라\n'
 '함은 아래의 경우 중 하나에 해당하는 때를 말한다.- 가) 해당 관절의 운동범위 합계가 정상운동범위의\n'
 '- 3/4 이하로 제한된 경우\n'
 '- 나) 근전도 검사상 불완전한 손상(incomplete\n'
 '- injury)소견이 있으면서 도수근력검사(MMT)에\n'
 '- 서 근력이 3등급(fair)인 경우\n'
 '11) “가관절주)이 남아 뚜렷한 장해를 남긴 때”라 함은\n'
 '상완골에 가관절이 남은 경우 또는 요골과 척골의 2'),
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
 'indexing': {'chunk_id': 'chunk_000574',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
