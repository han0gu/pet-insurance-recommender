from langchain_core.documents import Document

chunk = Document(
    page_content=('- 병\n'
 '- 할 수 있습니다.\n'
 '제4조("골절철심제거술"의 정의)\uf000 이 특별약관에 있어서 "골절철심제거술"이라함은 【별표7】(골절철심제거 수술- 려\n'
 '- 분류표)에서 정한 골절철심제거수술 대상 "수가코드"에 해당하는 경우를 말하며\n'
 '- 동\n'
 '- 해당 산정 기준일자는 치료개시일(해당 상병의 진료를 위하여 최초로 내원(입원\n'
 '- 물\n'
 '- 을 포함합니다)한 날을 말합니다)로 합니다.\n'
 '- \uf000 제1항의 "골절철심제거술"은 의료법 제3조(의료기관)에서 정한 국내의 병원 또는'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_000333',
              'chunk_char_len': 254,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
