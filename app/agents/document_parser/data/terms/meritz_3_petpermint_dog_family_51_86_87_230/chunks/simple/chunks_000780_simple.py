from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 해당 관절의 운동범위 합계가 정상운동범위의 1/2 이하로 제한된 경우 나) 객관적 검사(스트레스 엑스선)상 10mm 이상의 '
 '동요관절(관절이 흔들리거나 움직이는 것)이 있 는 경우 다) 근전도 검사상 불완전한 손상(incomplete injury)소견이 있으면서 '
 '도수근력검사(MMT)에 서 근력이 2등급(poor)인 경우\n'
 '10) “관절 하나의 기능에 약간의 장해를 남긴 때”라 함은 아래의 경우 중 하나에 해당하는 때를 말한다. 가) 해당 관절의 운동범위 '
 '합계가 정상운동범위의 3/4 이하로 제한된 경우'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 219},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000780',
              'chunk_char_len': 283,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
