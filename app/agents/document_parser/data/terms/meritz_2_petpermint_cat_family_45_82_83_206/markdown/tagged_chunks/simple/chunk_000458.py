from langchain_core.documents import Document

chunk = Document(
    page_content=('관련 특별약관 일반조항」을 따르고,「반려동물 비용손해\n'
 '관련 특별약관 일반조항」에서 정하지 않은 사항은 보통약\n'
 '관을 따릅니다.165Ⅱ. 기타 특별약관1. 반려동물 특정 질병 보장제한부 인수 특별약관# 제1조(특별약관의 체결 및 효력)\uf000 이 '
 '특별약관은 보험계약(특별약관이 부가된 경우에는 특\n'
 '별약관을 포함합니다. 이하 같습니다)을 체결할 때 반려동\n'
 '물의 건강상태가 회사가 정한 기준에 적합하지 않은 경우\n'
 '또는 보험계약을 체결한 후 계약 전 알릴 의무 위반의 효과\n'
 '등으로 보장을 제한할 경우 보험계약자(이하 「계약자」라'),
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
 'indexing': {'chunk_id': 'chunk_000458',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
