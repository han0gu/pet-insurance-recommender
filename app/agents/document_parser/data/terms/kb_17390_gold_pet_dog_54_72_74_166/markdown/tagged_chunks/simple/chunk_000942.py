from langchain_core.documents import Document

chunk = Document(
    page_content=('- 른다.\n'
 '4) 뇌전증- \n'
 '# 가) “뇌전증”이라 함은 돌발적 뇌파이상을 나타내는 뇌질환으로 발작(경련, 의식장해 등)을 반복하는 것을 말한다.\n'
 '나) 뇌전증 발작의 빈도 및 양상은 지속적인 항뇌전증제(항경련제) 약물\n'
 '로도 조절되지 않는 뇌전증을 말하며, 진료기록에 기재되어 객관적\n'
 '으로 확인되는 뇌전증 발작의 빈도 및 양상을 기준으로 한다.- \n'
 '- 154 -- \n'
 '- 다) “심한 뇌전증 발작”이라 함은 월 8회 이상의 중증발작이 연 6개월\n'
 '- 이상의 기간에 걸쳐 발생하고, 발작할 때 유발된 호흡장애, 흡인성'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'head']},
 'indexing': {'chunk_id': 'chunk_000942',
              'chunk_char_len': 284,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
