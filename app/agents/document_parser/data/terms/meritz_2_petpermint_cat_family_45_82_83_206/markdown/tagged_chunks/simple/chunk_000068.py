from langchain_core.documents import Document

chunk = Document(
    page_content=('(효력회복))에서 정한 계약의 부활이 이루어진 경우 부활을\n'
 '청약한 날을 제5항의 청약일로 하여 적용합니다.# 제20조(청약의 철회)\uf000 일반금융소비자인 계약자는 보험증권을 받은 날부터 '
 '15\n'
 '일 이내에 그 청약을 철회할 수 있습니다. 다만, 회사가 건\n'
 '강상태 진단을 지원하는 계약, 보험기간이 90일 이내인 계\n'
 '약 또는 전문금융소비자가 체결한 계약은 청약을 철회할 수\n'
 '없습니다.# 【일반금융소비자】# 전문금융소비자가 아닌 계약자를 말합니다.# 【전문금융소비자】보험계약에 관한 전문성, 자산규모 등에 비추어 '
 '보험계'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000068',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
