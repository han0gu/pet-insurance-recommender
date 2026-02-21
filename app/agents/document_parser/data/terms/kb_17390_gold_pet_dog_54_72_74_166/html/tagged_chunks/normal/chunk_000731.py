from langchain_core.documents import Document

chunk = Document(
    page_content=(". 정상분만, 치과질환</h1><br><p id='50' data-category='list' "
 "style='font-size:14px'>별<br>제4조(입원의 정의와 장소)<br>약<br>\uf000 이 특별약관에 있어서 "
 '"입원"이라 함은 병원 또는 의원의 의사, 치과의사 또는 한<br>관<br>의사 면허를 가진 자(이하 "의사"라 합니다)에 의하여 '
 '제1조(보험금의 지급사유)에<br>서 정한 지급사유의 치료가 필요하다고 인정한 경우로서 자택 등에서 치료가 곤란<br>하여 의료기관에 '
 '입실하여 의사의 관리하에 치료에 전념하는 것을'),
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
 'indexing': {'chunk_id': 'chunk_000731',
              'chunk_char_len': 292,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
