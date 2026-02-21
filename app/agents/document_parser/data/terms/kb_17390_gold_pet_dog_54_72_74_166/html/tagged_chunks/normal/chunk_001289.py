from langchain_core.documents import Document

chunk = Document(
    page_content=(". 정상분만, 치과질환</p><br><h1 id='126' style='font-size:14px'>제4조(입원의 정의와 "
 "장소)</h1><br><h1 id='127' style='font-size:14px'>\uf000 이 특별약관에</h1><br><p "
 'id=\'128\' data-category=\'paragraph\' style=\'font-size:14px\'>있어서 "입원"이라 '
 "함은 병원 또는 의원의 의사, 치과의사 또는 한</p><br><p id='129' data-category='list' "
 "style='font-size:14px'>의사"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'definition', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001289',
              'chunk_char_len': 298,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
