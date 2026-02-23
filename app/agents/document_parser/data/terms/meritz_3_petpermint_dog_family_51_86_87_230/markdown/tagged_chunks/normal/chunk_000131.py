from langchain_core.documents import Document

chunk = Document(
    page_content=('멸시효가 완성되어 보험금 등을 지급받지 못할 수 있습니\n'
 '다.제42조(약관의 해석)83# \uf000 회사는 신의성실의 원칙에 따라 공정하게 약관을 해석하\n'
 '여야 하며 계약자에 따라 다르게 해석하지 않습니다.# 【 신의성실의 원칙 】권리의 행사와 의무의 이행은 신의와 성실을 가지고 행동\n'
 '하여 상대방의 신뢰와 기대를 배반하여서는 안된다는 원칙\n'
 '(「민법」제2조 제1항)# 【 민법 제2조(신의성실) 제1항 】① 권리의 행사와 의무의 이행은 신의에 좇아 성실히 하\n'
 '여야 한다.\uf000 회사는 약관의 뜻이 명백하지 않은 경우에는 계약자에게'),
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
 'indexing': {'chunk_id': 'chunk_000131',
              'chunk_char_len': 288,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
