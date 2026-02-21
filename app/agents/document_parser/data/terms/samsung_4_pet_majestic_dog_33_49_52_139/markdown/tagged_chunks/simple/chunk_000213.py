from langchain_core.documents import Document

chunk = Document(
    page_content=('회사가 건강상태 진단을 지원하는 계약, 보험기간이 90일 이내인 계약 또는 전문금융\n'
 '소비자가 체결한 계약은 청약을 철회할 수 없습니다.# <용어풀이># [전문금융소비자]보험계약에 관한 전문성, 자산규모 등에 비추어 '
 '보험계약에 따른 위험감수능력이 있는 자로서,\n'
 '국가, 지방자치단체, 한국은행, 금융회사, 주권상장법인 등을 포함하며 「금융소비자 보호에 관한\n'
 '법률」제2조(정의) 제9호에서 정하는 전문금융소비자를 말합니다.\n'
 '[일반금융소비자]'),
    metadata={'source_doc': {'total_pages': 107},
 'doc': {'doc_type': 'terms',
         'file_name': 'samsung_4_pet_majestic_dog_33_49_52_139.pdf',
         'insurer_code': 'samsung',
         'product_code': '4',
         'product_name': '(장기)무배당 삼성화재 펫보험 위풍댕댕',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000213',
              'chunk_char_len': 243,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
