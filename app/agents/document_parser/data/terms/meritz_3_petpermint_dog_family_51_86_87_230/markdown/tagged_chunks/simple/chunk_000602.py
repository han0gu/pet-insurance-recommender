from langchain_core.documents import Document

chunk = Document(
    page_content=('“언어청력검사, 임피던스 청력검사, 청성뇌간반응검\n'
 '사(ABR), 이음향방사검사”등을 추가실시 후 장해를\n'
 '평가한다.# 다. 귓바퀴의 결손- 1) “귓바퀴의 대부분이 결손된 때”라 함은 귓바퀴의 연\n'
 '- 골부가 1/2이상 결손된 경우를 말한다.\n'
 '- 2) 귓바퀴의 연골부가 1/2 미만 결손이고 청력에 이상이\n'
 '- 없으면 외모의 추상(추한 모습)장해로만 평가한다.\n'
 '# 라. 평형기능의 장해1) “평형기능에 장해를 남긴 때”라 함은 전정기관 이\n'
 '상으로 보행 등 일상생활이 어려운 상태로 아래의 평\n'
 '형장해 평가항목별 합산점수가 30점 이상인 경우를'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000602',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
