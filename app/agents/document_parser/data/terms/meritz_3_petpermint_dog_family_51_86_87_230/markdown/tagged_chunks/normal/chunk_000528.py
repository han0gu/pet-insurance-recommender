from langchain_core.documents import Document

chunk = Document(
    page_content=('- 법인의 업무를 집행하는 그 밖의 기관)또는 이들의 법\n'
 '- 정대리인의 고의\n'
 '- ② 전쟁, 혁명, 내란, 사변, 테러, 폭동, 소요, 노동쟁\n'
 '- 의, 기타 이들과 유사한 사태\n'
 '- ③ 지진, 분화, 홍수, 해일 또는 이와 비슷한 천재지변\n'
 '- ④ 핵연료물질 또는 핵연료물질에 의하여 오염된 물질의\n'
 '- 방사성, 폭발성, 그 밖의 유해한 특성 또는 이들의 특\n'
 '- 성에 의한 사고\n'
 '- ⑤ 제4호 이외의 방사선을 쬐는 것 또는 방사능 오염'),
    metadata={'source_doc': {'total_pages': 180},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_3_petpermint_dog_family_51_86_87_230.pdf',
         'insurer_code': 'meritz',
         'product_code': '3',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000528',
              'chunk_char_len': 239,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
