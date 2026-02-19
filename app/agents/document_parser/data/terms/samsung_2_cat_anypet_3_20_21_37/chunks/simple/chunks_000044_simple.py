from langchain_core.documents import Document

chunk = Document(
    page_content=('【전문금융소비자】 보험계약에 관한 전문성, 자산규모 등에 비추어 보험계약에 따른 위험감수능력이 있는 자로서, 국가, 지방자치단체, '
 '한국은행, 금융회사, 주권상장법인 등을 포함 하며 「금융소비자보호에 관한 법률」 제2조(정의) 제9호에서 정하는 전문금융소비자를 '
 '말합니다. 【일반금융소비자】 전문금융소비자가 아닌 계약자를 말합니다.\n'
 '제1항에도 불구하고 청약한 날부터 30일이 초과된 계약은 청약을 철회할 수 없습니다.\n'
 '【관련법규】'),
    metadata={'source_doc': {'total_pages': 35},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_2_cat_anypet_3_20_21_37.pdf',
         'insurer_code': 'samsung',
         'product_code': '2',
         'product_name': '(일반)반려묘보험 애니펫',
         'total_pages': 35,
         'page': 11},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000044',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
