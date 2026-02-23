from langchain_core.documents import Document

chunk = Document(
    page_content=('이<br>병<br>특별약관은 그때부터 효력이 없습니다.<br>\uf000 이 특별약관의 계약자는 전환대상계약의 계약자와 동일하여야 '
 "합니다.</p><h1 id='88' style='font-size:16px'>제2조(제출서류)</h1><br><p id='89' "
 "data-category='paragraph' style='font-size:14px'>상<br>\uf000 이 특별약관에 가입하고자 "
 '하는 계약자는 모든 피보험자 또는 모든 보험수익자의 "<br>해<br>소득세법 시행규칙 별지 제38호 서식에 의한 장애인증명서의 원본 '
 '또는'),
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
 'indexing': {'chunk_id': 'chunk_001423',
              'chunk_char_len': 287,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.25}},
)
