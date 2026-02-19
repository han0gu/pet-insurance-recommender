from langchain_core.documents import Document

chunk = Document(
    page_content=('평균순음역치가 70dB이상인 경우에 해당되어, 50cm 이상의 거리에서는 보통의 말소리를 알아듣지 못하 는 경우를 말한다.\n'
 '5) 순음청력검사를 실시하기 곤란하거나(청력의 감소가 의 심되지만 의사소통이 되지 않는 경우, 만 3세 미만의 소아 포함) 검사결과에 '
 '대한 검증이 필요한 경우에는 “언어청력검사, 임피던스 청력검사, 청성뇌간반응검 사(ABR), 이음향방사검사”등을 추가실시 후 장해를 '
 '평가한다.\n'
 '다. 귓바퀴의 결손'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 180},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000639',
              'chunk_char_len': 234,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
