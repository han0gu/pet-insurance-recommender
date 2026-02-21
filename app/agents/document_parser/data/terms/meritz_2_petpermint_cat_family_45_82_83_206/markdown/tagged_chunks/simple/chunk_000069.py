from langchain_core.documents import Document

chunk = Document(
    page_content=('약에 따른 위험감수능력이 있는 자로서, 국가, 지방자치\n'
 '단체, 한국은행, 금융회사, 주권상장법인 등을 포함하며\n'
 '「금융소비자보호에 관한 법률」제2조(정의) 제9호에서\n'
 '정하는 전문금융소비자를 말합니다.\uf000 제1항에도 불구하고 청약한 날부터 30일(만 65세 이상의\n'
 '계약자가 전화를 이용하여 체결한 계약은 45일로 합니다)이\n'
 '초과된 계약은 청약을 철회할 수 없습니다.\n'
 '\uf000 청약철회는 계약자가 전화로 신청하거나, 철회의사를 표\n'
 '시하기 위한 서면, 전자우편, 휴대전화 문자메시지 또는 이\n'
 '에 준하는 전자적 의사표시(이하‘서면 등’이라 합니다)를'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Cat&Family보험 다이렉트2601',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000069',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
