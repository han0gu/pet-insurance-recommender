from langchain_core.documents import Document

chunk = Document(
    page_content=('- 정한 수의사에 의하여 백내장/녹내장으로 인해 수술이 필요하다고 인정된 경우로\n'
 '- 서 수정체 유화술, 수정체 낭내 적출술, 녹내장 수술 등의 수술을 말합니다.\n'
 '- \uf000 제1항 제3호에서 "이물제거(내시경)" 및 "이물제거(구토유도약물)"이라 함은 아\n'
 '- 래의 의료행위를 말합니다.\n'
 '- 1. "이물제거(내시경)"이란 반려동물의 위장 등 내부의 이물질을 제거하기 위하\n'
 '- 여 수술을 동반하지 않고 내시경 및 내시경포셉을 이용하여 비침습적으로 시\n'
 '- 행하는 의료행위를 말합니다.'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'other', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_000593',
              'chunk_char_len': 264,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
