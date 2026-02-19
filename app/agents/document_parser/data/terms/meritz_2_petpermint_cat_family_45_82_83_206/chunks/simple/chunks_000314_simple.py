from langchain_core.documents import Document

chunk = Document(
    page_content=('① 계약자, 피보험자, 이들의 가족 또는 사용인의 고의 또는 중대한 과실 ② 지진, 분화, 해일, 홍수 또는 이와 유사한 자연재해로 생긴 '
 '손해 ③ 전쟁, 외국의 무력행사, 혁명, 내란, 사변, 폭동, 소 요, 기타 이들과 유사한 사태 ④ 핵연료물질 또는 핵연료물질에 의하여 '
 '오염된 물질의 방사성, 폭발성, 그 밖의 유해한 특성 또는 이들의 특성에 의한 사고 ⑤ 제4호 이외의 방사선을 쬐는 것 또는 방사능 '
 '오염\n'
 '【핵연료물질】\n'
 '사용된 연료를 포함합니다.\n'
 '【핵연료물질에 의하여 오염된 물질】\n'
 '원자핵분열 생성물을 포함합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 111},
 'term_type': 'special',
 'clause': {'clause_type': 'exclusion', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000314',
              'chunk_char_len': 290,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
