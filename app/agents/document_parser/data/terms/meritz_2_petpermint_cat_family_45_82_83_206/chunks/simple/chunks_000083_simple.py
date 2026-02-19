from langchain_core.documents import Document

chunk = Document(
    page_content=('【일반금융소비자】\n'
 '전문금융소비자가 아닌 계약자를 말합니다.\n'
 '【전문금융소비자】\n'
 '보험계약에 관한 전문성, 자산규모 등에 비추어 보험계 약에 따른 위험감수능력이 있는 자로서, 국가, 지방자치 단체, 한국은행, 금융회사, '
 '주권상장법인 등을 포함하며 「금융소비자보호에 관한 법률」제2조(정의) 제9호에서 정하는 전문금융소비자를 말합니다.'),
    metadata={'source_doc': {'total_pages': 160},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_2_petpermint_cat_family_45_82_83_206.pdf',
         'insurer_code': 'meritz',
         'product_code': '2',
         'product_name': '무배당 펫퍼민트 Puppy&Family보험 다이렉트2601',
         'total_pages': 160,
         'page': 64},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000083',
              'chunk_char_len': 183,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
