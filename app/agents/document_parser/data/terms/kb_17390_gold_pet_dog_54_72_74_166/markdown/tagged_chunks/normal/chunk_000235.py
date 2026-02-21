from langchain_core.documents import Document

chunk = Document(
    page_content=('. 전지에 임한 자, 침몰한 선박 중에 있던 자, 추락한 항공기 중에 있던 자,기타 사망의 원인이 될 위난을 당한 자의 생사가 전쟁 종지 '
 '후 또는 선박의침몰, 항공기의 추락, 기타 위난이 종료한 후 1년간 분명하지 않은 때에도제1항과 같습니다.\uf000 "호스피스·완화의료 '
 '및 임종과정에 있는 환자의 연명의료 결정에 관한 법률"에- 따른 연명의료중단등결정 및 그 이행으로 피보험자가 사망하는 경우 연명의료중'),
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
 'indexing': {'chunk_id': 'chunk_000235',
              'chunk_char_len': 223,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
