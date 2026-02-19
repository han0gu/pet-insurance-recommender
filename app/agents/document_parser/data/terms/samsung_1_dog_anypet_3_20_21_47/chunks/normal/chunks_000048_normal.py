from langchain_core.documents import Document

chunk = Document(
    page_content=('【전문금융소비자】 보험계약에 관한 전문성, 자산규모 등에 비추어 보험계약에 따른 위험감수능력이 있는 자로서, 국가, 지방자치단체, '
 '한국은행, 금융회사, 주권상장법인 등을 포함 하며 「금융소비자보호에 관한 법률」 제2조(정의) 제9호에서 정하는 전문금융소비자를 '
 '말합니다.\n'
 '【일반금융소비자】 전문금융소비자가 아닌 계약자를 말합니다.\n'
 '제1항에도 불구하고 청약한 날부터 30일이 초과된 계약은 청약을 철회할 수 없습니다.\n'
 '【관련법규】'),
    metadata={'source_doc': {'total_pages': 45},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_1_dog_anypet_3_20_21_47.pdf',
         'insurer_code': 'samsung',
         'product_code': '1',
         'product_name': '(일반)반려견보험 애니펫',
         'total_pages': 45,
         'page': 11},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000048',
              'chunk_char_len': 238,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
