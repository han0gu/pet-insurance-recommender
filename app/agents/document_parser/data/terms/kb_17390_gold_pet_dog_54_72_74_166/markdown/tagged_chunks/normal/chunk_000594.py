from langchain_core.documents import Document

chunk = Document(
    page_content=('- 행하는 의료행위를 말합니다.\n'
 '- 2. "이물제거(구토유도약물)"이라 함은 반려동물의 위장 등 내부의 이물질을 제\n'
 '- 거하기 위하여 수술 및 내시경을 동반하지 않고 구토유발을 목적으로 한 약물\n'
 '- 을 이용한 의료행위를 말합니다.\n'
 '- \uf000 제1항 제4호에서 "창상"이란 찢어진 상처를 말하며 "교상"이란 물린 상처를 말합\n'
 '- 니다.\n'
 '- \uf000 제1항 제5호에서 "특정약물치료Ⅱ"라 함은 수의사가 반려동물의 상해 또는 질병\n'
 '- 의 치료를 직접적인 목적으로 아포퀠(Apoquel) 등의 JAK inhibitor(Janus kinase'),
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
 'indexing': {'chunk_id': 'chunk_000594',
              'chunk_char_len': 293,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
