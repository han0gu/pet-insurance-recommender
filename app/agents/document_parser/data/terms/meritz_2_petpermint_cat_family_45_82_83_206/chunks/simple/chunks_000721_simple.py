from langchain_core.documents import Document

chunk = Document(
    page_content=('. 6) “발가락에 뚜렷한 장해를 남긴 때”라 함은 첫째 발 가락의 경우에 중족지관절과 지관절의 굴신(굽히고 펴 기)운동범위 합계가 정상 '
 '운동가능영역의 1/2 이하가 된 경우를 말하며, 다른 네 발가락에 있어서는 중족지 관절의 신전운동범위만을 평가하여 정상운동범위의 1/2 '
 '이하로 제한된 경우를 말한다. 7) 한 발가락에 장해가 생기고 다른 발가락에 장해가 발 생한 경우, 지급률은 각각 적용하여 합산한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 198},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['joint', 'other']},
 'indexing': {'chunk_id': 'chunk_000721',
              'chunk_char_len': 226,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
