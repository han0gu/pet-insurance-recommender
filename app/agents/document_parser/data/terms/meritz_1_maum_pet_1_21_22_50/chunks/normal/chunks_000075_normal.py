from langchain_core.documents import Document

chunk = Document(
    page_content=('【일반금융소비자】전문금융소비자가 아닌 계약자를 말합니다. 【전문금융소비자】보험계약에 관한 전문성, 자산규모 등에 비추어 보험계약에 따른 '
 '위험감수능력이 있는 자로서, 국가, 지방자치단체, 한국은행, 금융회사, 주권상장법 인 등을 포함하며 「금융소비자보호에 관한 '
 '법률」제2조(정의) 제9호에서 정하는 전 문금융소비자를 말합니다.'),
    metadata={'source_doc': {'total_pages': 50},
 'doc': {'doc_type': 'terms',
         'file_name': 'meritz_1_maum_pet_1_21_22_50.pdf',
         'insurer_code': 'meritz',
         'product_code': '1',
         'product_name': '메리츠 마음든든 반려동물보험',
         'total_pages': 50,
         'page': 12},
 'term_type': 'basic',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000075',
              'chunk_char_len': 181,
              'embedding_model': 'solar-embedding-1-large',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
