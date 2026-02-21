from langchain_core.documents import Document

chunk = Document(
    page_content=("id='1' data-category='list' style='font-size:14px'>신체의 단면영상을 얻거나 3차원적인 입체영상을 "
 '얻는 영상진단법<br>\uf000 제1항 제2호에서 "백내장/녹내장수술"이라 함은 제1조(보험금의 지급사유)에서<br>정한 수의사에 '
 '의하여 백내장/녹내장으로 인해 수술이 필요하다고 인정된 경우로<br>서 수정체 유화술, 수정체 낭내 적출술, 녹내장 수술 등의 수술을 '
 '말합니다.<br>\uf000 제1항 제3호에서 "이물제거(내시경)" 및 "이물제거(구토유도약물)"이라 함은 아</p><br><p '
 "id='2'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive', 'eye']},
 'indexing': {'chunk_id': 'chunk_001028',
              'chunk_char_len': 294,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
