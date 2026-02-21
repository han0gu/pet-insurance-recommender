from langchain_core.documents import Document

chunk = Document(
    page_content=('관련 특별약관 일반조항」을 따르고,「반려동물 비용손해\n'
 '관련 특별약관 일반조항」에서 정하지 않은 사항은 보통약\n'
 '관을 따릅니다.173# Ⅱ. 배상책임 관련 특별약관# 배상책임 관련 특별약관 일반조항# 제1조(목적)이 특별약관은 계약자와 회사 사이에 '
 '피보험자가 법률상의\n'
 '배상책임을 부담함으로써 입은 손해에 대한 위험을 보장하\n'
 '기 위하여 체결됩니다.# 제2조(용어의 정의)이 특별약관에서 사용되는 용어의 정의는, 이 특별약관의\n'
 '다른 조항에서 달리 정의되지 않는 한 다음과 같습니다.# \uf000 계약관계 관련 용어| 용어 | 정의 |'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000476',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
