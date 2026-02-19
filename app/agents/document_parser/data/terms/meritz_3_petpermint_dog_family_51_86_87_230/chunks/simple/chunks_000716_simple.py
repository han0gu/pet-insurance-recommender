from langchain_core.documents import Document

chunk = Document(
    page_content=('다. 귓바퀴의 결손\n'
 '1) “귓바퀴의 대부분이 결손된 때”라 함은 귓바퀴의 연 골부가 1/2이상 결손된 경우를 말한다. 2) 귓바퀴의 연골부가 1/2 미만 '
 '결손이고 청력에 이상이 없으면 외모의 추상(추한 모습)장해로만 평가한다.\n'
 '라. 평형기능의 장해\n'
 '1) “평형기능에 장해를 남긴 때”라 함은 전정기관 이 상으로 보행 등 일상생활이 어려운 상태로 아래의 평 형장해 평가항목별 합산점수가 '
 '30점 이상인 경우를 말한다.\n'
 '항목 | 내 용 | 점수\n'
 '검사 소견 | 양측 전정기능 소실 | 14\n'
 '양측 전정기능 감소 | 10'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 205},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other', 'other']},
 'indexing': {'chunk_id': 'chunk_000716',
              'chunk_char_len': 285,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.85}},
)
