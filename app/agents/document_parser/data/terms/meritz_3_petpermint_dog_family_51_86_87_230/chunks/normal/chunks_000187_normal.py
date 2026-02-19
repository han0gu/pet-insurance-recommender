from langchain_core.documents import Document

chunk = Document(
    page_content=('자기 부담금 | 보험사고로 인하여 발생한 손해에 대하여 계약 자 또는 피보험자가 부담하는 일정 금액을 말 합니다.\n'
 '보험금 분담 | 이 계약에서 보장하는 위험과 같은 위험을 보 장하는 다른 계약(공제계약을 포함합니다)이 있을 경우 비율에 따라 손해를 '
 '보상합니다.\n'
 '공제계약 | 공제(미래에 발생할 수 있는 경제적 불안을 제 거하기 위해 공동으로 재산을 준비하여 두는 제도) 사업을 실시하는 경영 주체와 '
 '공제 계약 자 사이에 체결되는 계약을 말합니다.\n'
 '\uf000 지급금과 이자율 관련 용어\n'
 '용어 | 정의'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 90},
 'term_type': 'special',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000187',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
