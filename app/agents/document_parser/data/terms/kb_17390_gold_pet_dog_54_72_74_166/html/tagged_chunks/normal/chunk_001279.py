from langchain_core.documents import Document

chunk = Document(
    page_content=('. 다만, 질<br>아래와 같이 반려동물 위탁비용이 지급된 최종입원일부터 180일이 경과하도록 퇴 병<br>원없이 계속 입원중인 경우에는 '
 "반려동물 위탁비용이 지급된 최종입원일의 그 다<br>음날을 퇴원일로 봅니다.<br>반</p><br><p id='113' "
 "data-category='paragraph' style='font-size:14px'>예 시</p><br><p id='114' "
 "data-category='paragraph' style='font-size:14px'>반려동물 위탁비용이</p><br><figure "
 "id='115'"),
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
 'indexing': {'chunk_id': 'chunk_001279',
              'chunk_char_len': 299,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
