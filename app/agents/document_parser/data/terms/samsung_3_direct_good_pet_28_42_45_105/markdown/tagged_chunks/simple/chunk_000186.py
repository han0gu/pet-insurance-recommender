from langchain_core.documents import Document

chunk = Document(
    page_content=('- 약관의 부활이 이루어진 경우 부활을 청약한 날을 제6항의 청약일로 하여 적용합니\n'
 '- 다.\n'
 '# 제20조 (청약의 철회)① 계약자는 보험증권을 받은 날부터 15일 이내에 그 청약을 철회할 수 있습니다. 다만,\n'
 '회사가 건강상태 진단을 지원하는 계약, 보험기간이 90일 이내인 계약 또는 전문금융\n'
 '소비자가 체결한 계약은 청약을 철회할 수 없습니다.<용어풀이># [전문금융소비자]보험계약에 관한 전문성, 자산규모 등에 비추어 보험계약에 '
 '따른 위험감수능력이 있는 자로서,'),
    metadata={'source_doc': {'total_pages': 78},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_3_direct_good_pet_28_42_45_105.pdf',
         'insurer_code': 'samsung',
         'product_code': '3',
         'product_name': '(장기)무배당 삼성화재 다이렉트 '
                         '착한펫보험(강아지)',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000186',
              'chunk_char_len': 259,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
