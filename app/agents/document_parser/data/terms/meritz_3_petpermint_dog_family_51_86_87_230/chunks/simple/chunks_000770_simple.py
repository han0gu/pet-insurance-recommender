from langchain_core.documents import Document

chunk = Document(
    page_content=('가) 해당 관절의 운동범위 합계가 정상운동범위의 3/4 이하로 제한된 경우 나) 근전도 검사상 불완전한 손상(incomplete '
 'injury)소견이 있으면서 도수근력검사(MMT)에 서 근력이 3등급(fair)인 경우\n'
 '11) “가관절주)이 남아 뚜렷한 장해를 남긴 때”라 함은 상완골에 가관절이 남은 경우 또는 요골과 척골의 2 개뼈 모두에 가관절이 남은 '
 '경우를 말한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 217},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['joint']},
 'indexing': {'chunk_id': 'chunk_000770',
              'chunk_char_len': 206,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
