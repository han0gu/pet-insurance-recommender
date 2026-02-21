from langchain_core.documents import Document

chunk = Document(
    page_content=('관련 특별약관 일반조항」을 따르고,「반려동물 비용손해\n'
 '관련 특별약관 일반조항」에서 정하지 않은 사항은 보통약\n'
 '관을 따릅니다.1172. 펫퍼민트 반려견 입원의료비보장 특별약관# 제1조(보험금의 지급사유)# ① 고급형\uf000 회사는 보험기간 중에 '
 '보험증권에 기재된 반려동물에게\n'
 '질병 또는 상해가 발생하여 그 치료를 직접적인 목적으로\n'
 '수의사법 제2조(정의)에서 정한 국내 동물병원(이하 「동물\n'
 '병원」이라 합니다)에 입원하여 수의사법 제2조(정의)에서\n'
 '정한 수의사(이하 「수의사」라 합니다)에게 치료를 받은'),
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
 'indexing': {'chunk_id': 'chunk_000260',
              'chunk_char_len': 276,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
