from langchain_core.documents import Document

chunk = Document(
    page_content=('. 가) 언어평가상 자음정확도가 50%미만인 경우 나) 언어평가상 표현언어지수 25 미만인 경우 9) “말하는 기능에 약간의 장해를 남긴 '
 '때”라 함은 아 래의 경우 중 하나 이상에 해당되는 때를 말한다. 가) 언어평가상 자음정확도가 75%미만인 경우 나) 언어평가상 '
 '표현언어지수 65 미만인 경우 10) 말하는 기능의 장해는 1년 이상 지속적인 언어치료를 시행한 후 증상이 고착되었을 때 평가하며, '
 '객관적인 검사를 기초로 평가한다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 183},
 'term_type': 'special',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['head']},
 'indexing': {'chunk_id': 'chunk_000655',
              'chunk_char_len': 240,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
