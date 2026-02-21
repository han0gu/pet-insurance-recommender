from langchain_core.documents import Document

chunk = Document(
    page_content=('. 단, "특정처치(이물제거)"로 인한 주요치료보<br>험금은 "이물제거(내시경)" 횟수와 "이물제거(구토유도약물)" 횟수를 '
 "합산하여<br>연간 2회를 한도로 합니다.</p><p id='200' data-category='paragraph' "
 "style='font-size:14px'>112 KB 금쪽같은 펫보험(강아지)(무배당)(26.01)</p><br><p id='201' "
 "data-category='paragraph' style='font-size:14px'>예 시 주요치료보험금의 계산<br>[주요치료보험금"),
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
 'indexing': {'chunk_id': 'chunk_000997',
              'chunk_char_len': 286,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
