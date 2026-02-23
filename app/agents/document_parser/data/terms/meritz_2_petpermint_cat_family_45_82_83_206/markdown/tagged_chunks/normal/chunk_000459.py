from langchain_core.documents import Document

chunk = Document(
    page_content=('등으로 보장을 제한할 경우 보험계약자(이하 「계약자」라\n'
 '합니다)의 청약과 보험회사의 승낙으로 보험계약(이하 「계\n'
 '약」이라 합니다)에 부가하여 이루어집니다.\n'
 '\uf000 제1항에 따라 이 특약을 부가할 때 반려동물의 과거 병\n'
 '력과 수의학적으로 또는 경험통계적으로 인과관계가 유의성\n'
 '있게 확인된 경우 등과 같이 회사가 정한 기준에 따라 직접\n'
 '관련이 있는 특정질병으로 제한하며, 부담보 설정 범위 및\n'
 '사유를 계약자에게 설명하여 드립니다.\n'
 '\uf000 이 특별약관의 보장개시일은 보통약관 제26조(제1회 보'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000459',
              'chunk_char_len': 269,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
