from langchain_core.documents import Document

chunk = Document(
    page_content=('제2조(보험금을 지급하지 않는 사유)\n'
 '\uf000 회사는 다음 중 어느 한 가지로 보험금 지급사유가 발생 한 때에는 보험금을 지급하지 않습니다.\n'
 '① 계약자, 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실 ② 지진, 분화, 해일, 홍수 또는 이와 유사한 자연재해로 생긴 '
 '손해 ③ 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소 요, 기타 이들과 유사한 사태 ④ 핵연료물질 또는 핵연료물질에 의하여 '
 '오염된 물질의 방사성, 폭발성, 그 밖의 유해한 특성 또는 이들의'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 180,
         'page': 124},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000358',
              'chunk_char_len': 265,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.85}},
)
