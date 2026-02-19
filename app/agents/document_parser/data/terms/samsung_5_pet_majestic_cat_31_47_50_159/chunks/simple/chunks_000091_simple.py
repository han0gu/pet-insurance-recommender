from langchain_core.documents import Document

chunk = Document(
    page_content=('<용어풀이>\n'
 '[전문금융소비자]\n'
 '보험계약에 관한 전문성, 자산규모 등에 비추어 보험계약에 따른 위험감수능력이 있는 자로서, 국 가 , 지방자치단체, 한국은행, 금융회사, '
 '주권상장법인 등을 포함하며 「금융소비자 보호에 관한 법 률」제2조(정의) 제9호에서 정하는 전문금융소비자를 말합니다. [일반금융소비자]\n'
 '전문금융소비자가 아닌 계약자를 말합니다.\n'
 '② 제1항에도 불구하고 청약한 날부터 30일(다만, 65세 이상의 계약자가 전화를 이용하 여 계약을 체결한 경우 45일)이 초과된 계약은 '
 '청약을 철회할 수 없습니다.\n'
 '<관련법규>'),
    metadata={'source_doc': {'total_pages': 129},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_5_pet_majestic_cat_31_47_50_159.pdf',
         'insurer_code': 'samsung',
         'product_code': '5',
         'product_name': '(장기)무배당 삼성화재 펫보험 의기냥냥',
         'total_pages': 107,
         'page': 39},
 'term_type': 'basic',
 'clause': {'clause_type': 'other', 'risk_domains': ['other']},
 'indexing': {'chunk_id': 'chunk_000091',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'llm',
              'tag_confidence': 0.95}},
)
