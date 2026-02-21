from langchain_core.documents import Document

chunk = Document(
    page_content=('요청할 수 있습니다.<br>\uf000 회사는 제1항에 따른 개선이 완료될 때까지 계약의 효력을 정지할 수 있습니다.<br>\uf000 '
 '회사는 이 계약의 중요사항과 관련된 범위 내에서는 보험기간 중 또는 회사에서<br>정한 보험금 청구서류를 접수한 날부터 1년 이내에는 '
 "언제든지 피보험자의 회계<br>장부를 열람할 수 있습니다.</p><br><p id='31' data-category='paragraph' "
 "style='font-size:16px'>제22조(특별약관의 무효)</p><br><h1 id='32'"),
    metadata={'source_doc': {'total_pages': 113},
 'doc': {'doc_type': 'terms',
         'file_name': 'kb_17390_gold_pet_dog_54_72_74_166.pdf',
         'insurer_code': 'kb',
         'product_code': '17390',
         'product_name': '[일반보험] KB반려행복펫보험',
         'total_pages': 1,
         'page': 1},
 'term_type': 'unknown',
 'clause': {'clause_type': 'claim', 'risk_domains': ['digestive']},
 'indexing': {'chunk_id': 'chunk_001231',
              'chunk_char_len': 274,
              'embedding_model': 'solar-embedding-1-large',
              'tag_type': 'normal',
              'tag_method': 'rule',
              'tag_confidence': 0.55}},
)
