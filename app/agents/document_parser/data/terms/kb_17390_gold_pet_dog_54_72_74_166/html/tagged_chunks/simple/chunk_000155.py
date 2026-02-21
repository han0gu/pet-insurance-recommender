from langchain_core.documents import Document

chunk = Document(
    page_content=('. 보장성 상품: 일반금융소비자가 「상법」 제640조에 따른 보험증권을 받은<br>날부터 15일과 청약을 한 날부터 30일 중 먼저 '
 "도래하는 기간</p><p id='193' data-category='paragraph' "
 "style='font-size:14px'>제2조(정의) 이 법에서 사용하는 용어의 뜻은 다음과 같다.</p><br><p id='194' "
 "data-category='paragraph' style='font-size:14px'>9"),
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
 'indexing': {'chunk_id': 'chunk_000155',
              'chunk_char_len': 249,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'simple',
              'tag_method': 'rule',
              'tag_confidence': 0.8}},
)
