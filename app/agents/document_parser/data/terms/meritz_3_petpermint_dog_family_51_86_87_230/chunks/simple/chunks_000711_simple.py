from langchain_core.documents import Document

chunk = Document(
    page_content=('. 12) “눈꺼풀에 뚜렷한 결손을 남긴 때”에 해당하는 경 우에는 추상(추한 모습)장해를 포함하여 장해를 평가 한 것으로 보고 '
 '추상(추한 모습)장해를 가산하지 않 는다. 다만, 안면부의 추상(추한 모습)은 두 가지 장 해평가 방법 중 피보험자에게 유리한 것을 '
 '적용한다.'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 204},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['eye', 'skin']},
 'indexing': {'chunk_id': 'chunk_000711',
              'chunk_char_len': 151,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
