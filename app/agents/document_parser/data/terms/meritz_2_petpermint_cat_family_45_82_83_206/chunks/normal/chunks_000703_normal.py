from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 해당 관절의 운동범위 합계가 정상운동범위의 1/4이하로 제한된 경우\n'
 '나) 인공관절이나 인공골두를 삽입한 경우 다) 객관적 검사(스트레스 엑스선)상 15mm 이상의 동요관절(관절이 흔들리거나 움직이는 것)이 '
 '있 는 경우 라) 근전도 검사상 완전손상(complete injury) 소견 이 있으면서 도수근력검사(MMT)에서 근력이 “1 '
 '등급(Trace)"인 경우\n'
 '9) “관절 하나의 기능에 뚜렷한 장해를 남긴 때”라 함 은 아래의 경우 중 하나에 해당하는 때를 말한다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 194},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000703',
              'chunk_char_len': 263,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
