from langchain_core.documents import Document

chunk = Document(
    page_content=(". 우체국, 신협, 새마을금고 등이 공제계약을 취급합니다.</p><br><p id='126' "
 "data-category='paragraph' style='font-size:14px'>상</p><br><p id='127' "
 "data-category='paragraph' style='font-size:14px'>해</p><br><p id='128' "
 "data-category='paragraph' style='font-size:16px'>제6조(특별약관의 소멸)<br>\uf000 회사는 "
 '제1조(보험금의 지급사유)에서 정한 반려동물장례비용지원금을 지급한'),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'coverage', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001124',
              'chunk_char_len': 300,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
