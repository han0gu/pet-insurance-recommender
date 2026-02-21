from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 해당 관절의 운동범위 합계가 정상운동범위의\n'
 '3/4 이하로 제한된 경우194- 나) 객관적 검사(스트레스 엑스선)상 5mm 이상의 동요관\n'
 '- 절(관절이 흔들리거나 움직이는 것)이 있는 경우\n'
 '- 다) 근전도 검사상 불완전한 손상(incomplete\n'
 '- injury)소견이 있으면서 도수근력검사(MMT)에\n'
 '- 서 근력이 3등급(fair)인 경우\n'
 '- 11) 동요장해 평가 시에는 정상측과 환측을 비교하여 증\n'
 '- 가된 수치로 평가한다.\n'
 '- 12) "가관절주)이 남아 뚜렷한 장해를 남긴 때"라 함은'),
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
 'indexing': {'chunk_id': 'chunk_000585',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
