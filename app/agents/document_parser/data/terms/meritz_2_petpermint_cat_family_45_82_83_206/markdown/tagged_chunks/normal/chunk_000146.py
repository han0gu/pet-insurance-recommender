from langchain_core.documents import Document

chunk = Document(
    page_content=('85# \uf000 지급사유 관련 용어| 용어 | 정의 |\n'
 '| --- | --- |\n'
 '| 상해 | 보험기간 중에 발생한 급격하고도 우연한 외래 의 사고로 반려동물이 입은 상해를 말합니다. 유독가스 또는 유독물질을 우연히 '
 '일시적으로 흡입, 흡수 또는 섭취한 결과로 발생하는 중독 증상을 포함합니다. 그러나 세균성 음식물 중 독과 상습적으로 흡입, 흡수 또는 '
 '섭취한 결과 로 생긴 중독증상은 이에 포함되지 않습니다. |\n'
 '| 질병 | 상해를 제외한 상병을 모두 포함합니다. |'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000146',
              'chunk_char_len': 257,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
