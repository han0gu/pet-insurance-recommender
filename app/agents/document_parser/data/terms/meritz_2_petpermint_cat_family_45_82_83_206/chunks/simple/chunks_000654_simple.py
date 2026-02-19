from langchain_core.documents import Document

chunk = Document(
    page_content=('6) 부정교합은 위턱(상악)과 아래턱(하악)의 부조화로 윗니(상악치아)와 아랫니(하악치아)가 전방 및 측 방으로 맞물림에 제한이 있는 '
 '상태를 말한다. 7) “말하는 기능에 심한 장해를 남긴 때”라 함은 아래 의 경우 중 하나 이상에 해당되는 때를 말한다. 가) 언어평가상 '
 '자음정확도가 30%미만인 경우 나) 전실어증, 운동성실어증(브로카실어증)으로 의사 소통이 불가한 경우 8) “말하는 기능에 뚜렷한 장해를 '
 '남긴 때”라 함은 아래의 경우 중 하나 이상에 해당되는 때를 말한 다'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 183},
 'term_type': 'special',
 'clause': {'clause_type': 'other', 'risk_domains': ['dental', 'head']},
 'indexing': {'chunk_id': 'chunk_000654',
              'chunk_char_len': 267,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
